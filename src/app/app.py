from flask import Flask, flash, request, redirect, url_for, Request, Response, send_from_directory
from werkzeug.utils import secure_filename
from werkzeug.wrappers.response import Response as wResponse
import os
from src.denoising import DenoisingModelInterface, DenoisingCV2, DenoisingUNet
import cv2
from typing import Union
from enum import Enum

UPLOAD_FOLDER = os.path.abspath("data")
MAX_FILE_SIZE = 16  # MB
ALLOWED_EXTENSIONS = set(['png'])

class Algorithm(str, Enum):
    CV2 = "CV2"
    UNet = "UNet"

def getUNet():
    try:
        return DenoisingUNet()
    except:
        return DenoisingModelInterface()

ALGORITHMS = {
    Algorithm.CV2 : DenoisingCV2(),
    Algorithm.UNet : getUNet()  # set UNet if cuda available, else set Interface to raise error
}

class DenoiserWebApp(Flask):
    """
    A class used to run denoiser of your choice in a simple web app

    Methods
    -------
    run(port=5000, debug=False)
    """

    def __init__(
            self, import_name: str, upload_directory: str, max_file_size: int,
            denoiser: DenoisingModelInterface, allowed_extensions: set = {'png'}, debug: bool = False
        ):
        """
        Parameters
        ----------
        import_name : str
            The name of the module, just write '__name__' if using directly
        upload_directory : str
            The absolute path to the director in which images should be saved
        max_file_size : int
            The maximum size of a uploaded file (in MB)
        denoiser : DenoisingModelInterface
            Object implementing DenoisingModelInterface interface used to denoise the image
        allowed_extensions : set, optional
            A set of allowed extensions for uploaded files (default is only 'png')
        debug : bool, optional
            A flag for displaying additional debugging information (default is False)
        """

        super().__init__(import_name)
        self.config['UPLOAD_FOLDER'] = upload_directory
        self.config['MAX_CONTENT_LENGTH'] = max_file_size * 10**6
        self.denoiser: DenoisingModelInterface = denoiser
        self.allowed_extensions: set = allowed_extensions
        self.DEBUG: bool = debug
        prepare_denoiser_web_app(self)

    def allowed_file(self, filename: str) -> bool:
        return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in self.allowed_extensions

    def denoise_rw(self, filename: str, suffix: str = '_denoised') -> str:
        img = cv2.imread(os.path.join(self.config['UPLOAD_FOLDER'], filename))
        if self.DEBUG:
            cv2.imshow('img1', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        img = self.denoiser.denoise(img)
        if self.DEBUG:
            cv2.imshow('img1', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        file, ext = filename.rsplit('.', 1)
        filename_denoised = '{}{}.{}'.format(file, suffix, ext)
        cv2.imwrite(os.path.join(self.config['UPLOAD_FOLDER'], filename_denoised), img)
        return filename_denoised

def prepare_denoiser_web_app(app: DenoiserWebApp) -> DenoiserWebApp:
    @app.route('/', methods=['GET', 'POST'])
    def upload_file() -> Union[str, wResponse]:
        if request.method == 'POST':
            # check if the post request has the file part
            if 'file' not in request.files:
                flash('No file part')
                return redirect(request.url)
            algorithm = request.form['algorithm']
            # Set selected denoising algorithm
            app.denoiser = ALGORITHMS.get(algorithm)
            file = request.files['file']
            # If the user does not select a file, the browser submits an
            # empty file without a filename.
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file and app.allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                filename_denoised = app.denoise_rw(filename)
                return redirect(url_for('download_file', name=filename_denoised))
        return '''
<!doctype html>
<head>
    <title>Main denoiser page</title>
</head>
<body style="background-color:#e4e4d9;
padding: 1em;
width: 100%;
height: 100vh;
box-sizing: border-box;
background-image:linear-gradient(175deg, rgba(0,0,0,0) 95%, #8da389 95%),
                 linear-gradient( 85deg, rgba(0,0,0,0) 95%, #8da389 95%),
                 linear-gradient(175deg, rgba(0,0,0,0) 90%, #b4b07f 90%),
                 linear-gradient( 85deg, rgba(0,0,0,0) 92%, #b4b07f 92%),
                 linear-gradient(175deg, rgba(0,0,0,0) 85%, #c5a68e 85%),
                 linear-gradient( 85deg, rgba(0,0,0,0) 89%, #c5a68e 89%),
                 linear-gradient(175deg, rgba(0,0,0,0) 80%, #ba9499 80%),
                 linear-gradient( 85deg, rgba(0,0,0,0) 86%, #ba9499 86%),
                 linear-gradient(175deg, rgba(0,0,0,0) 75%, #9f8fa4 75%),
                 linear-gradient( 85deg, rgba(0,0,0,0) 83%, #9f8fa4 83%),
                 linear-gradient(175deg, rgba(0,0,0,0) 70%, #74a6ae 70%),
                 linear-gradient( 85deg, rgba(0,0,0,0) 80%, #74a6ae 80%);
">
<div style="text-align:center;width:80%;">
    <h1 style="font-family:ubuntu;">Super Denoiser 3000</h1>
    <form method=post enctype=multipart/form-data>
        <input type=file name=file>
        <select name="algorithm">
            <option value="CV2">CV2</option>
            <option value="UNet">UNet</option>
        </select>
        <input type=submit value=Upload>
    </form>
</div>
</body>
'''

    @app.route('/uploads/<name>')
    def download_file(name: str) -> Response:
        return send_from_directory(app.config["UPLOAD_FOLDER"], name)
    
    return app

if __name__ == "__main__":
    app = DenoiserWebApp(__name__, UPLOAD_FOLDER, MAX_FILE_SIZE, DenoisingCV2())
    app.run(debug=True, port=8000)
