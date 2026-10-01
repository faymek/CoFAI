# Poetry Environment Management

**Adding Packages**
```sh
poetry add <package>
```

# Git Workflow

The default branch is `dev`. Features should be developed until complete before being merged into the `main` branch.

Each contributor should create a new branch. For example:
```sh
git checkout -b dev/feature1
```

**How to submit:**
```sh
git push origin dev/feature2
```

After pushing, create a Pull Request on GitHub to merge your changes into the `dev` branch.

# Code Integration

For new projects, please place them under `examples/project/`. Once initial results are achieved, reusable components can be merged into the `cofai/` directory.

# Documentation

Pull requests targeting `dev` or `main`, and pushes to either branch, build and check the documentation. Only `main` publishes to the documentation website; `dev` does not deploy. GitHub Pages environment protection should continue to allow only `main`. To publish documentation changes, review them on `dev` and merge them into `main` with the corresponding release. The workflow can also be run manually; manual runs on `dev` remain build-only.

The documentation is built using [Zensical](https://zensical.org/) — the successor to [MkDocs Material](https://squidfunk.github.io/mkdocs-material/), currently in development.

To get started, install the required packages:

```sh
poetry install --only docs --no-root
```

Then, launch the local documentation server with:

```sh
poetry run zensical serve
```

Once the server is running, you can view the documentation at [http://localhost:8000](http://localhost:8000).
