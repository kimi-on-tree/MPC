# Setting up your virtual environment


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