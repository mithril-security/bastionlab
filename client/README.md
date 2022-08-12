# BlindAI Client

BastionAI Client is a python library to create client applications for BastionAI Server (Mithril-security's confidential training server). 

**If you wish to know more about BastionAI, please have a look to the project [Github repository](https://github.com/mithril-security/bastionai/).**

## Installation

### Using pip
```bash
$ pip install bastionais
```
### Install from source
Install development version (modifications to the code are immediately reflecting without the need to rebuild)
```bash
$ make install
```

Build a wheel from source
```bash
$ make build
```
The wheel wil be located in `./dist`.

## Usage

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under [Apache 2.0 License.](https://github.com/mithril-security/blindai/blob/master/LICENSE)

The project uses the "Intel SGX DCAP Quote Validation Library" for attestation verification, See [Intel SGX DCAP Quote Validation Library License](https://github.com/intel/SGXDataCenterAttestationPrimitives/blob/master/License.txt)
