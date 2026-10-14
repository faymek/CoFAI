# Plain-DETR components

These internal modules are extracted from the DINOv3 detection implementation added in PR #15 (commit `d41e8026`). Copyright notices and the upstream license are retained. They implement the ViT-L/16 Plain-DETR head used by the COCO test plan; CoFAI supplies the four intermediate feature maps.

The extraction excludes the official ViT, image-backbone adapter, window wrappers, distributed utilities, training metrics, alternate absolute-position decoder, and reparameterized model. The head constructs its encoder, relative-position decoder and prediction layers directly. Parameter names and inference operations are preserved for the supplied checkpoint.
