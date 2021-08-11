### Development notes

Poetry usage notes:

1. `poetry init` inside the root folder to start poetry project

2. change the `pyproject.toml` file about dependencies. I think that the '^' decorator (e.g. python = '^3.8') is the most useful. See poetry website for details.

3. `poetry update` to generate the lock file and install all the dependencies to a venv.

4. `poetry run python script.py` to run script.py in the new venv. OR `poetry shell` to activate the new venv. One useful way I found to use it is `poetry run python -c 'import deeptissue'` to make sure that I did `poetry init` in the right folder.

5. `poetry config repositories.testpypi https://test.pypi.org/legacy/` to define the testpypi directory as the "testpypi" repository for publishing.

6. `poetry version [BUMP_SIZE]` to bump version. (e.g. `poetry version patch`)

7. `poetry publish --build -r testpypi -u [USERNAME] -p [PASSWORD]` to publish. Alternatively, run `poetry build` then `poetry publish`. The username and password parameters would be asked for interactively if not passed in through command line.
