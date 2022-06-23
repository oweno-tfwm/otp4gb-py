# python-otp4gb

This is a port of the [OTP4GB][OTP4GB] tool written by Tom Forth at Open Innovations.

[OTP4GB]: https://github.com/odileeds/OTP4GB

## Preparation and dependencies

Make sure you have a working Java environment, and have Maven installed.

Run the setup script to install dependencies.

```
python setup.py
```

This will run maven and download the osmconvert64 executable for Windows.

## Running

### `prepare.py`

This script prepares the osm directory.

command line options

  `-F` Force overwrite of existing files

Environment config

  `OSMCONVERT_DOCKER` - if set, will use a dockerised version of osmconvert (useful to run on macos, where there appears to be no native osmconvert)