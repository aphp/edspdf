# Contributing to EDS-PDF

We welcome contributions ! There are many ways to help. For example, you can:

1. Help us track bugs by filing issues
2. Suggest and help prioritise new functionalities
3. Help us make the library as straightforward as possible, by simply asking questions on whatever does not seem clear to you.

## Development installation

To be able to run the test suite and develop your own pipeline, you should clone the repo and install it locally. We use the [`hatch`](https://hatch.pypa.io/) package manager to manage the project.


<div class="termy">

```console
color:gray # Clone the repository and change directory
$ git clone ssh://git@github.com/aphp/edspdf.git
---> 100%

color:gray # Ensure hatch is installed, preferably via pipx
$ pipx install hatch

$ cd edspdf

color:gray # Enter a shell to develop / test the project. This will install everything required in a virtual environment. You can also `source` the path shown by hatch.
$ hatch shell
$ ...
$ exit  # when you're done
```

</div>

To make sure the pipeline will not fail because of formatting errors, we added pre-commit hooks using the `pre-commit` Python library. To use it, simply install it:

<div class="termy">

```console
$ pre-commit install
```

</div>

The pre-commit hooks defined in the [configuration](https://github.com/aphp/edspdf/blob/main/.pre-commit-config.yaml) will automatically run when you commit your changes, letting you know if something went wrong.

The hooks only run on staged changes. To force-run it on all files, run:

<div class="termy">

```console
$ pre-commit run --all-files
---> 100%
color:green All good !
```

</div>

## Proposing a merge request

At the very least, your changes should :

- Be well-documented ;
- Pass every tests, and preferably implement its own ;
- Follow the style guide.

### Testing your code

We use the Pytest test suite.

The following command will run the test suite. Writing your own tests is encouraged !

```shell
pytest
```

Should your contribution propose a bug fix, we require the bug be thoroughly tested.

### Style Guide

We use [Black](https://github.com/psf/black) to reformat the code. While other formatter only enforce PEP8 compliance, Black also makes the code uniform. In short :

> Black reformats entire files in place. It is not configurable.

Moreover, the CI/CD pipeline enforces a number of checks on the "quality" of the code. To wit, non black-formatted code will make the test pipeline fail. We use `pre-commit` to keep our codebase clean.

Refer to the [development install tutorial](#development-installation) for tips on how to format your files automatically.
Most modern editors propose extensions that will format files on save.

### Documentation

Make sure to document your improvements, both within the code with comprehensive docstrings,
as well as in the documentation itself if need be.

We use `MkDocs` for EDS-PDF's documentation. You can view your changes with

<div class="termy">

```console
color:gray # Run the documentation
$ hatch run docs:serve
```

</div>

Go to [`localhost:8000`](http://localhost:8000) to see your changes. MkDocs watches for changes in the documentation folder
and automatically reloads the page.
