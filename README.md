# cableloss

[![PyPi Version][pypi ver image]][pypi ver link]
[![Build Status][travis image]][travis link]
[![Coverage Status][coveralls image]][coveralls link]
[![License Badge][license image]][LICENSE.txt]

[cableloss][] is a Python 3.3+ module that calculates the cable loss for a given
cable type and length.

## Requirements

- [numpy][]

## Contributing

Contributions are welcome! To contribute please:

1. Fork the repository
2. Create a feature branch
3. Code
4. Submit a [pull request][]

### Virtualenv

Use the following commands to create a Python 3.9.9 virtualenv using [pyenv][]
and [pyenv-virtualenv][], install the requirements in the virtualenv named
`sdfascii`, and list the available [Invoke][] tasks.

```bash
$ pyenv virtualenv 3.9.9 sdfascii
$ pyenv activate sdfascii
$ pip install -r requirements.txt
$ inv -l
```

### Submitting Pull Requests

[keysight][] is developed using [Scott Chacon][]'s [GitHub Flow][]. To
contribute, fork [sdfascii][], create a feature branch, and then submit
a pull request.  [GitHub Flow][] is summarized as:

- Anything in the `master` branch is deployable
- To work on something new, create a descriptively named branch off of
  `master` (e.g., `new-oauth2-scopes`)
- Commit to that branch locally and regularly push your work to the same
  named branch on the server
- When you need feedback or help, or you think the brnach is ready for
  merging, open a [pull request][].
- After someone else has reviewed and signed off on the feature, you can
  merge it into master.
- Once it is merged and pushed to `master`, you can and *should* deploy
  immediately.

# License

[cableloss][] is released under the MIT license. Please see the
[LICENSE.txt] file for more information.

[cableloss]: https://github.com/questrail/cableloss
[coveralls image]: http://img.shields.io/coveralls/questrail/cableloss/master.svg
[coveralls link]: https://coveralls.io/r/questrail/cableloss
[LICENSE.txt]: https://github.com/questrail/cableloss/blob/develop/LICENSE.txt
[license image]: http://img.shields.io/pypi/l/cableloss.svg
[numpy]: http://www.numpy.org
[pull request]: https://help.github.com/articles/using-pull-requests
[pypi ver image]: http://img.shields.io/pypi/v/cableloss.svg
[pypi ver link]: https://pypi.python.org/pypi/cableloss
[python standard library]: https://docs.python.org/2/library/
[travis image]: http://img.shields.io/travis/questrail/cableloss/master.svg
[travis link]: https://travis-ci.org/questrail/cableloss
