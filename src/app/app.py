from flask import Flask, flash, request, redirect, url_for, Request, Response, send_from_directory
from werkzeug.utils import secure_filename
from werkzeug.wrappers.response import Response as wResponse
import os
from src.denoising import DenoisingModelInterface, DenoisingCV2, DenoisingUNet
import cv2
from typing import Union

UPLOAD_FOLDER = os.path.abspath("data")
MAX_FILE_SIZE = 16  # MB
ALLOWED_EXTENSIONS = set(['png'])

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
        <title>Upload new File</title>
        <h1>Upload new File</h1>
        <form method=post enctype=multipart/form-data>
        <input type=file name=file>
        <input type=submit value=Upload>
        </form>
        '''

    @app.route('/uploads/<name>')
    def download_file(name: str) -> Response:
        return send_from_directory(app.config["UPLOAD_FOLDER"], name)
    
    return app

if __name__ == "__main__":
    app = DenoiserWebApp(__name__, UPLOAD_FOLDER, MAX_FILE_SIZE, DenoisingCV2())
    app.run(debug=True, port=8000)
