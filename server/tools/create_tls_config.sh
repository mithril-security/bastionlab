#!/bin/sh

if [ $# -eq 0 ];
then   
    echo -e "\nPlease provide bin directory!\n"
    exit 1
fi

TLS_DIR=$1/tls
mkdir -p ${TLS_DIR}

openssl req -newkey rsa:2048 -nodes -keyout ${TLS_DIR}/host_server.key  \
		-x509 -days 365 -out ${TLS_DIR}/host_server.pem \
		-subj "/C=FR/CN=bastionai-srv"
