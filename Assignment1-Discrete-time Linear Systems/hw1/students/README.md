# HW1 

To start with the assignment, download the code of the assignment in a empty folder on your machine. A good structure would be to have a folder called `mpc_assignments` which looks like this 

```
mpc_assignments
            ├── hw1 # this is the one for your first assignment
            |    ├── .venv       # your virtual environment goes here (see next section)
            |    ├── controller.py
            │    ├── linear_car_model.py
            │    ├── README.md
            │    ├── requirements.txt
            │    ├── sim_env.py
            │    └── task1.py
            ├── hw2
            ├── hw3
            ├── hw4

```

# Setting up your virtual environment

To create your `.venv` where all your packages for this assignment live, your should enter your folder

```
cd /PATH_TO_YOUR_ASSIGNEMNT/mpc_assignments/hw1/
```

Start by creating a virtual environment with `uv`

```bash
uv venv
```

you should see something like 

```bash
Using CPython 3.10.12 interpreter at: /usr/bin/python3
Creating virtual environment at: .venv
Activate with: source .venv/bin/activate
```

At this point, activate the it (choose the right command depending on your machine):

```bash
source .venv/bin/activate      # macOS/Linux
```
```bash
.venv\Scripts\activate       # Windows (cmd)
```
```bash
.venv\Scripts\Activate.ps1   # Windows (PowerShell)
```

And install the libraries for this exercise

```bash
uv pip install -r requirements.txt
```

You should see something like 

```bash
Resolved 25 packages in 584ms
Prepared 6 packages in 4.97s
Installed 25 packages in 39ms
 + casadi==3.8.0
 + cffi==2.1.1
 + clarabel==0.11.1
 + contourpy==1.3.2
 + control==0.10.2
 + cvxpy==1.7.5
 + cycler==0.12.1
 + fonttools==4.63.0
 + jinja2==3.1.6
 + joblib==1.5.3
 + kiwisolver==1.5.0
 + markupsafe==3.0.3
 + matplotlib==3.10.9
 + mosek==11.2.3
 + numpy==2.2.6
 + osqp==1.1.3
 + packaging==26.3
 + pillow==12.3.0
 + pycparser==3.0
 + pyparsing==3.3.2
 + python-dateutil==2.9.0.post0
 + scipy==1.15.3
 + scs==3.2.11
 + setuptools==84.0.0
 + six==1.17.0
```

Verify what got installed

```bash
uv pip list
```

If you later want to deactivate your virtual environment, you can just type 

```
deactivate
```

**Note**: Some IDE already start your venv for you, such as VScode or Pycharm.

For VScode you can look here :  [Find your venv in VScode](https://code.visualstudio.com/docs/python/environments#:~:text=for%20future%20compatibility.-,Select%20an%20environment,-To%20use%20a) 

For Pycharm you can look here :  [Find your venv in Pycharm](https://www.jetbrains.com/help/pycharm/installing-uninstalling-and-reloading-interpreter-paths.html)