import os
from flask import Flask, render_template, request, send_from_directory, redirect
import time
from subprocess import call
import cv2

app = Flask(__name__)

APP_ROOT=os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return render_template("drag_drop.html")

@app.route("/upload",methods=['POST', "GET"])
def upload():
    if request.method=="POST":
        folder = str(int(time.time()*1000000))
        target = os.path.join(APP_ROOT,'static/uploads')
        target = os.path.join(target, folder)
        if not os.path.isdir(target):
            os.mkdir(target)

        for files in request.files.getlist("file"):
            filename = files.filename
            file_ext = filename.split(".")[-1]
            destination = "/".join([target, 'input.'+file_ext])
            
            files.save(destination)
            
            call(['python', 'static/main.py', folder, file_ext])
            global path
            path = {
                    "img1":"uploads/"+folder+"/input."+file_ext,
                    # "img2":"uploads/"+folder+"/noise_free_preprocessed."+file_ext,
                    "img2":"uploads/"+folder+"/preprocessed-resized."+file_ext,
                    "img3":"uploads/"+folder+"/ROIsegmented."+file_ext,
                    "img4":"uploads/"+folder+"/words_level_segmented."+file_ext,
                    "img5":"uploads/"+folder+"/before_correction."+file_ext,
                    "img6":"uploads/"+folder+"/after_correction."+file_ext,
                    }

        return "Done"
    else:
        return redirect("/")


@app.route("/uploaded",methods=['POST', 'GET'])
def uploaded():
    global path
    try:
        localpath=path
        path=None
        return render_template("index123.html", details=localpath)
    except NameError:
        return redirect("/")


if __name__ == "__main__":
    app.run(port = 5000, debug = True)
