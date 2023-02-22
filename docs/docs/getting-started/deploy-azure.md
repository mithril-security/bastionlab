# Deploy on Azure
____________________________________________

Get started and **Deploy BastionLab Server**

## Pre-requisites
___________________________________________

### Requirements

To deploy **BastionLab Server**, ensure the following requirements are satisfied:

- [Azure account](https://portal.azure.com/)
    - If you would like to follow along but don't have an [Azure account](https://docs.microsoft.com/en-us/azure/guides/developer/azure-developer-guide#understanding-accounts-subscriptions-and-billing), make sure to create a [free](https://azure.microsoft.com/free/?ref=microsoft.com&utm_source=microsoft.com&utm_medium=docs&utm_campaign=visualstudio) one before you start.

If deploying with **Azure CLI** in your local environment, then you will need to install:

- [Azure CLI](https://learn.microsoft.com/en-us/cli/azure/)
    - To get the latest version of Azure CLI for your system, you can go to: [https://www.python.org/downloads/](https://www.python.org/downloads/)
    - or use a one line command for **debian-based** distros:
  ```bash
  curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
  ```

> Be mindful when creating and deleting resources as some of the samples presented in this article may result in charges, especially if certain deployment settings are chosen or if your application is left running for an extended period. To avoid incurring any unexpected costs, please be sure to review the documentation and billing pages closely prior to any deployments.

## Deploying with Azure Portal
_____________________________________________

First you will need to login to your [Azure Portal](https://portal.azure.com/) and create a **resource group**.
### Create a resource group
<img src="../../assets/az-resource-group.png" alt="resource group" width="90%" />

Then go to the **Container instances** section in your azure portal.
### Setup basic configuration
<img src="../../assets/az-container-instance.png" alt="resource group" width="90%" />

### Setup networking
Let the **public type networking** option, set your DNS label and set the port to 50056.

<img src="../../assets/az-container-instance-net.png" alt="resource group" width="90%" />

Leave all other settings as their defaults, then select Review + create.

### Create container instance
Review the settings and create the container instance.

<img src="../../assets/az-container-instance-create.png" alt="resource group" width="90%" />

### Uploading authentication public keys
<img src="../../assets/az-container-instance-keys.png">

## Deploying with Azure CLI
______________________________________________

First you will need to login with the command:
```bash
az login
```
### Basic Configuration Setup
Make sure to set *resource group name*, *app name*, *location* and the *docker image* variables according to your needs.

##### Set variables (example values)
```bash
resourceGroupName="docker-bastionlab"
appName="bastionlab-docker-$RANDOM"
location="eastus"
bastionLabImage="mithrilsecuritysas/bastionlab:latest"
```
##### Create a Resource Group
```bash
az group create --name $resourceGroupName --location $location
```

### Deploy BastionLab Server as a Container App
```bash
az container create \
        --name $appName \
        --resource-group $resourceGroupName \
        --image $bastionLabImage \
        --dns-name-label $appName \
        --ports 50056
```

### Show FQDN and Provisioning State

To display the container's fully qualified domain name (FQDN) and its provisioning state, you can run:

```bash
az container show \
	--resource-group $resourceGroupName \
	--name $appName \
	--query "{FQDN:ipAddress.fqdn,ProvisioningState:provisioningState}" \
	--out table
```

If the `ProvisioningState` is **Succeeded**, congratulations! You have deployed succesfully your application in a running Docker container on Azure Cloud.

```bash
FQDN                               ProvisioningState
---------------------------------  -------------------
<fqdn>				   Succeeded
```
