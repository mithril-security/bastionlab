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

First you will need to login to your [Azure Portal](https://portal.azure.com/) and create a *resource group*.
![](../../assets/RG-Azure-870x586.png)

## Deploying with Azure CLI
______________________________________________

First you will need to login with the command:
```bash
az login
```
### Basic Configuration Setup
Make sure to set *resource group name*, *app name*, *location* and *app service plan name* variables according to your needs.

##### Set variables (example values)
```bash
resourceGroupName="docker-RG"
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
```bash
az container show \
	--resource-group $resourceGroupName \
	--name $appName \
	--query "{FQDN:ipAddress.fqdn,ProvisioningState:provisioningState}" \
	--out table
```

