from app import DenoiserWebApp
from denoising import DenoisingCV2, DenoisingHybrid
import os

if __name__ == "__main__":
    app = DenoiserWebApp(
        __name__,
        upload_directory=os.path.abspath("data"),
        max_file_size=10,  # MB
        allowed_extensions={'png','jpg','jpeg'},
        debug=False
    )
    app.run(debug=True, port=8000)
