Installation
============

You can install the latest stable version from PyPI::

    pip install simulstream

Or, to install from source::

    git clone https://github.com/hlt-mt/simulstream.git
    cd simulstream
    pip install .

Please notice that these commands will only install the basic functionalities of the repository,
i.e. the WebSocket server and client. Additionally, you have to install the dependencies required
by the `speech processor` that you want to use. The repository comes with examples of speech
processors that rely on e.g. Transformers models or on the NVIDIA Canary model. You can install
the required dependencies for them by specifying the corresponding selector when installing
the repository. For instance, using Canary speech processors requires the ``canary`` selector.
But you can create your custom speech processor and install the corresponding dependencies.

Also the evaluation comes with additional dependencies, which can be installed with the ``eval``
selector. Be careful as the metrics include COMET, which has dependencies on Transformers and
other libraries that can run on conflict with those required by your speech processor.

As an example, if you want to install the ``simulstream`` package with Canary speech processors
and the evaluation package, run::

    pip install simulstream[canary,eval]

For development (with docs and testing tools)::

    pip install .[dev]
