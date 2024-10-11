This is our Project of ISD class in topic of De belles lunettes 
Which is classify the face shape of users and match frame to the face shape.


*** PLEASE READ THIS INSTRUCTION ***
1. Recommend to create python virtual environment for testing this project.
   *to create virtual env please run "python -m venv your-env-name"
   *and activate the virtual env please run "./your-env-name/Scripts/activate.bat" (if you use Window use .bat if mac use .ps)
2. then make sure you are in the virtual env
   if you use VScode you can press Shift + Ctrl + P short and type "python: select Interpreter" then select your virtual env.
3. Make sure you are in the virtual env. (recommend to change terminal and run "pip list" to check that you are in the env if are inside the lib list will not that many.)
4. then you can clone our github project into your directory virtual env. (git clone https://github.com/DabielC/project_isd.git)
5. Please change directory also run this bash "cd project_isd"
   The file Structure should be like this :</br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;current_directory ___ </br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|_  project_isd</br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|_ rest of the files ... </br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|_  your-env-name</br>
7. install the libraries which this application require by run this "pip install -r isd_requirements.txt"
8. you need to download our model folder from this google drive link cause the models too large to push into github.
   https://drive.google.com/drive/folders/1Fc7bL91qameHX9uBq8YXcHNwFabe3h1p?usp=sharing
   then put on your virtual env directory (project_isd) also.
9. please run this to start api server ** fastapi run ./API/main.py --host "127.0.0.1" --port 8000
10. then you can start the web server from any port make sure that you are in the localhost.
11. Lastly Have fun :) !!
