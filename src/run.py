from app import DenoiserWebApp
from denoising import DenoisingCV2, DenoisingHybrid
import os

if __name__ == "__main__":
    device = "cuda"
    # device = "cpu"
    # denoiser = DenoisingHybrid(device=device)
    denoiser = DenoisingCV2()
    app = DenoiserWebApp(
        __name__,
        upload_directory=os.path.abspath("data"),
        max_file_size=10,  # MB
        denoiser=denoiser,
        allowed_extensions={'png','jpg','jpeg'},
        debug=False
    )
    app.run(debug=True, port=8000)
