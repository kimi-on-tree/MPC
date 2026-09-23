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
Activate the it (choose the right command depending on your machine):

```bash
source .venv/bin/activate      # macOS/Linux
```
```bash
.venv\Scripts\activate       # Windows (cmd)
```
```bash
.venv\Scripts\Activate.ps1   # Windows (PowerShell)
```

Install the libraries for this exercise

```bash
uv pip install -r requirements.txt
```

Verify what got installed

```bash
uv pip list
```

Note: Some IDE already start your venv for you, such as VScode or Pycharm.

For VScode you can look here :  [Find your venv in VScode](https://code.visualstudio.com/docs/python/environments#:~:text=for%20future%20compatibility.-,Select%20an%20environment,-To%20use%20a) 

For Pycharm you can look here :  [Find your venv in Pycharm](https://www.jetbrains.com/help/pycharm/installing-uninstalling-and-reloading-interpreter-paths.html)