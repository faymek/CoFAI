import os
import tempfile
import torch.nn as nn
import torch

from cofai.latent_codecs import FixedGaussianCodec
from cofai.latent_codecs.hm_sandwitch import HMCodecFFmpeg

from cofai.models.cavc.sandwich_with_feature import CondTCM
from cofai.models.cavc.videoprocessor import VideoReader, VideoWriterWrapper
from cofai.backbone.vgg import VGGBackbone

from cofai.layers.cavc_layers import MixFeatureProjector
from cofai.utils.utils_cavc import update_registered_buffers

from tqdm import tqdm
import cv2
from torchvision import transforms
import torchvision


class CAVCodec(nn.Module):
    def __init__(self,
                device='cuda',
                prompt_dim=128, 
                feature_dim=512, 
                compress_feature_dim=64,
                cond_tcm={},
                fixed_scale=1.0,
                ffmpeg_path='ffmpeg',
                ffmpeg_verbose=False
                ):
        """
        初始化编解码器。
        在这里进行耗时的模型加载（Load Weights）和组件初始化。
        """
        super().__init__()
        print("[CAVCodec] 正在初始化模型组件...")

        self.vgg = VGGBackbone(device)
        self.mix_projector = MixFeatureProjector(prompt_dim, feature_dim, compress_feature_dim)
        self.sandwich = CondTCM(**cond_tcm)
        self.gauss = FixedGaussianCodec(fixed_scale)
        self.ffmpeg = HMCodecFFmpeg(ffmpeg_path, ffmpeg_verbose)
        self.video_writer = VideoWriterWrapper()
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        self.device = device
        print("[ModelCodec] 初始化完成")
    
    def update(self, scale_table=None, force=False):
        updated = self.sandwich.update(scale_table, force)
        return updated
    
    def load_state_dict(self, state_dict, strict=True):
        update_registered_buffers(
            self.sandwich.gaussian_conditional,
            "sandwich.gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        incompatible_keys = super().load_state_dict(state_dict, strict=strict)
        return incompatible_keys

    def compress_video(self, reader, meta, codec_args: dict):
        """
        流程:
        1. 读取视频
        2. VGG 提取特征 (Branch 2 源数据)
        3. 模型处理图像 (Branch 3 源数据)
        4. 压缩特征 -> Branch 2 Bin
        5. 写临时视频 -> FFmpeg -> Branch 3 Bin
        """
        
        # 1. 初始化读取器
        # reader = VideoReader(video_path)
        fps = reader.fps
        size = (reader.width, reader.height)
        
        image_list = []   # 存储处理后的 Tensor
        feature_list = [] # 存储 VGG 特征
        
        # print(f"[ModelCodec] 正在处理: {video_path}")
        
        # 2. 遍历处理 (Feature & Preprocess)
        for frame_bgr in tqdm(reader, total=reader.total_frames, desc="Extracting"):
            
            # --- VGG 特征提取 ---
            # 对应: feat = extract_vgg_feature(...)
            mode = codec_args.get('mode', 2) 
            vgg_feature = self.vgg.extract(frame_bgr)
            feature = self.mix_projector.forward(mode, vgg_feature)

            feature_list.append(feature)
            
            # --- 图像预处理 (Model Preprocess) ---
            # 对应原代码步骤1: BGR转RGB并转换为[0,1]范围的tensor
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            input_tensor = self.transform(frame_rgb).unsqueeze(0).to(self.device)
            
            # 对应原代码步骤2: model.preprocess
            enhanced_tensor = self.sandwich.preprocess(input_tensor, feature)
            
            image_list.append(enhanced_tensor)

        reader.release()

        # 3. Branch 2: 特征压缩
        coded_layer2 = {}
        for i, feature in enumerate(feature_list):
            bin_data, feature_shape = self.gauss.compress(feature)
            coded_layer2[i] = {"strings": bin_data, "pstate": feature_shape}

        # 4. Branch 3: 视频流压缩
        coded_layer3 = {}
        tmp_dir = os.path.join("tmp_cavc")
        os.makedirs(tmp_dir, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, dir=tmp_dir) as tmp_file:
            tmp_video_path = tmp_file.name
        
        try:
            # 使用封装好的 Writer 将 Tensor 列表写回视频
            self.video_writer.write_batch(image_list, tmp_video_path, fps, size)
            
            # 使用 FFmpeg 压缩
            # hack
            out_path = os.path.join(tmp_dir, meta['seq_name'])
            video_bin = self.ffmpeg.compress(tmp_video_path, out_path)
            coded_layer3["video"] = {"strings": video_bin, "pstate": {}}

        finally:
            if os.path.exists(tmp_video_path):
                try:
                    os.remove(tmp_video_path)
                except:
                    pass

        return {
            "type": "layer_wise",
            "data": {
                "layer2": coded_layer2,
                "layer3": coded_layer3,
            }
        }

    def decompress_video(self, coded_data: dict, codec_args: dict):
        """
        解码流程:
        1. 解析输入字典 (Features Bin, Video Bin, Shapes)
        2. 解码特征流 -> Feature List (Branch 2)
        3. 解码视频流 -> 临时视频文件 -> VideoReader (Branch 3)
        4. 联合后处理 (PostProcess/Sandwich) -> 恢复/任务结果
        """
        
        # =================================================
        # 1. 准备数据与参数
        # =================================================
        assert coded_data["type"] == "layer_wise"
        coded_layer2 = coded_data["data"]["layer2"] 
        coded_layer3 = coded_data["data"]["layer3"]
        
        # 目标任务模式，例如 ["rec", "cls", "det"]
        target_modes = codec_args.get("mode", 2) 
        if isinstance(target_modes, str):
            target_modes = [target_modes]


        # =================================================
        # 2. Branch 2: 解码特征 (Features Decompression)
        # =================================================
        feature_list = []
        # 遍历每一帧的压缩包
        for idx, coded_frame in coded_layer2.items():
            # 使用 Gauss 组件解压
            # pstate is just feature here
            feature = self.gauss.decompress(coded_frame["strings"], coded_frame["pstate"])
            feature_list.append(feature)

        # =================================================
        # 3. Branch 3: 解码视频 (Video Decompression)
        # =================================================
        video_bin_data = coded_layer3["video"]["strings"] # 获取视频二进制流
        
        # 创建临时文件用于存放 ffmpeg 解码后的视频
        tmp_dir = os.path.join("tmp_cavc")
        os.makedirs(tmp_dir, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, dir=tmp_dir) as tmp_file:
            tmp_video_path = tmp_file.name

        reader = None
        results = []
        try:
            # 调用 FFmpeg 解码 (Bytes -> MP4 File)

            # NamedTemporaryFile creates an empty file; remove it so ffmpeg can write cleanly.
            try:
                os.remove(tmp_video_path)
            except OSError:
                pass

            ok = self.ffmpeg.decompress(video_bin_data, tmp_video_path)
            if (not ok) or (not os.path.exists(tmp_video_path)) or (os.path.getsize(tmp_video_path) == 0):
                raise ValueError(f"FFmpeg decompress failed: {tmp_video_path}")

            # 初始化 VideoReader 读取解码后的帧
            reader = VideoReader(tmp_video_path)
            
            # =================================================
            # 4. 联合后处理 (Post-Processing / Sandwich)
            # =================================================
            # 同时遍历 解码后的视频帧 和 解码后的特征
            for frame_bgr, feature in tqdm(zip(reader, feature_list), 
                                         total=len(feature_list), 
                                         desc="Decompressing"):
                
                # 预处理: BGR -> RGB -> Tensor (CHW) -> Device
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                input_tensor = self.transform(frame_rgb).unsqueeze(0).to(self.device)
                
                # 结果字典
                frame_result = {}
                
                # --- 调用 Sandwich 模型进行后处理/恢复 ---
                restored_tensor = self.sandwich.postprocess(input_tensor, feature)
                
                # # 根据任务模式封装结果
                # if 2 in target_modes:
                #     # 如果需要返回 Tensor 或 numpy，可以在这里转换
                frame_result["x_hat"] = torch.clamp(restored_tensor, min=0., max=1.)

                # torchvision.utils.save_image(restored_tensor, '/mnt/netdisk2/sunsf/Code/CoFAI-merge/exp/tmp/output.png')
                
                # 如果有其他任务 (如分类 cls, 检测 det)，可以在这里添加逻辑
                # 例如: 
                # if "cls" in target_modes:
                #     frame_result["x_cls"] = self.classifier(restored_tensor)
                # 事实上，视频重建的结果可以用于多种任务，只是根据引导模式不同有不同倾向性
                # 这里应该区分的是存帧重建结果还是特征重建结果

                # if "feat" in target_modes:
                #     frame_result["f_hat"] = feature
                
                results.append(frame_result)

        finally:
            # 资源清理
            if reader:
                reader.release()
            
            if os.path.exists(tmp_video_path):
                try:
                    os.remove(tmp_video_path)
                except OSError:
                    pass

        return results
