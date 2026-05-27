

# NEUROJACKSPLAT-V2-70M

NeuroJackSplat-V2-70M is an advanced 3D Reconstruction pipeline that uses **DINOv2** and **Structure from Motion (COLMAP)** to reconstruct geometry from "in-the-wild" images.

## Topics
`3d-reconstruction` `computer-vision` `gaussian-splatting` `zero-shot-learning` `dinov2` `point-cloud` `colmap` `pytorch`

---

## Engineering (Pipeline)
1. **Calibration:** Automatic spatial structuring with `pycolmap`.
2. **Feature Extraction:** Using vision transformers (DINOv2) to understand depth.
3. **Latent Space Mapping:** Generating depth maps with our Cost-Volume architecture.
4. **Rendering:** Interactive 3D visualization via [Plotly/WebGL].

## Installation
```bash
pip install torch pycolmap plotly pillow numpy

```

## Usage

To generate a 3D point cloud from your images:

```python
# Run the pipeline with your image directory
python main.py --images_dir ./data/your_scene

```

## License

This project is licensed under the MIT License.

---

## View live on Hugging Face:
https://huggingface.co/itsjakup/neurojacksplat-v2-70m/upload/main
