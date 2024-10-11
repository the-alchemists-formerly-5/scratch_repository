## Setting Up the Development Environment

This project uses Poetry for dependency management and virtual environment creation. Follow these steps to set up your development environment.

### Prerequisites

- Python 3.9 or higher

### Step 1: Install Poetry

If you haven't installed Poetry yet, you can do so by running:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Verify the installation:

```bash
poetry --version
```

If poetry is not in there

```bash
 curl -sSL https://install.python-poetry.org | python3.11 -
```

### Step 2: Clone the Project

```bash
git clone https://github.com/your-username/your-project-name.git
cd your-project-name
```

### Step 3: Configure Poetry (Optional)

To have Poetry create the virtual environment in your project directory (recommended for better IDE integration):

```bash
poetry config virtualenvs.in-project true
```

### Step 4: Install Dependencies

```bash
poetry install
```

This command creates a new virtual environment (if it doesn't exist yet) and installs all the dependencies specified in `pyproject.toml`.

### Using the Virtual Environment

There are two ways to run your code using the Poetry-managed virtual environment:

1. Activate the virtual environment:

   ```bash
   poetry shell
   ```

   Then run your commands as usual:

   ```bash
   python your_script.py
   # or
   jupyter notebook
   ```

2. Run commands directly using Poetry:

   ```bash
   poetry run python your_script.py
   # or
   poetry run jupyter notebook
   ```

### Managing Dependencies

- Add a new package:

  ```bash
  poetry add package_name
  ```

- Add a development dependency:

  ```bash
  poetry add --dev package_name
  ```

- Update dependencies:

  ```bash
  poetry update
  ```

- Remove a package:

  ```bash
  poetry remove package_name
  ```

### Deactivating the Virtual Environment

If you used `poetry shell`, you can exit it by typing:

```bash
exit
```

### Exporting Dependencies

To generate a `requirements.txt` file:

```bash
poetry export -f requirements.txt --output requirements.txt
```

### Recreating the Environment

If you need to recreate the environment:

```bash
poetry env remove --all
poetry install
```

Remember to add `.venv/` to your `.gitignore` file if you've configured Poetry to create the virtual environment in your project directory.
