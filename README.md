# cableloss

[![PyPi Version][pypi ver image]][pypi ver link]
[![Build Status][travis image]][travis link]
[![Coverage Status][coveralls image]][coveralls link]
[![License Badge][license image]][LICENSE.txt]

[cableloss][] is a Python 3.3+ module that calculates the cable loss for a given
cable type and length. Supported cable types include:

- RG-58
- RG-58/U
- LMR-195
- LMR-400

## Requirements

- [numpy][]

## Contributing

Contributions are welcome! To contribute please:

1. Fork the repository
2. Create a feature branch
3. Add code and tests
4. Pass lint and tests
5. Submit a [pull request][]

## Development Setup

### Development Setup Using pyenv

Use the following commands to create a Python 3.9.9 virtualenv using [pyenv][]
and [pyenv-virtualenv][], install the requirements in the virtualenv named
`cableloss`, and list the available [Invoke][] tasks.

```bash
$ pyenv virtualenv 3.9.9 cableloss
$ pyenv activate cableloss
$ pip install -r requirements.txt
$ inv -l
```

# License

[cableloss][] is released under the MIT license. Please see the
[LICENSE.txt][] file for more information.

[cableloss]: https://github.com/questrail/cableloss
[coveralls image]: http://img.shields.io/coveralls/questrail/cableloss/master.svg
[coveralls link]: https://coveralls.io/r/questrail/cableloss
[invoke]: https://www.pyinvoke.org/
[LICENSE.txt]: https://github.com/questrail/cableloss/blob/develop/LICENSE.txt
[license image]: http://img.shields.io/pypi/l/cableloss.svg
[numpy]: http://www.numpy.org
[pull request]: https://help.github.com/articles/using-pull-requests
[pyenv]: https://github.com/pyenv/pyenv
[pyenv-install]: https://github.com/pyenv/pyenv#installation
[pyenv-virtualenv]: https://github.com/pyenv/pyenv-virtualenv
[pypi ver image]: http://img.shields.io/pypi/v/cableloss.svg
[pypi ver link]: https://pypi.python.org/pypi/cableloss
[python standard library]: https://docs.python.org/2/library/
[travis image]: http://img.shields.io/travis/questrail/cableloss/master.svg
[travis link]: https://travis-ci.org/questrail/cableloss
