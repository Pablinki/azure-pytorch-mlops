// Platform resources for textclf. Deployed once (make provision); app.bicep is redeployed by CI.
targetScope = 'resourceGroup'

@description('Short lowercase base name (<= 10 chars)')
@minLength(3)
@maxLength(10)
param baseName string = 'textclf'

param location string = resourceGroup().location

var suffix = uniqueString(resourceGroup().id)
var storageName = toLower(take('st${baseName}${suffix}', 24)) // storage names: 3-24 chars, alphanumeric
var acrName = toLower('acr${baseName}${suffix}')
var kvName = 'kv-${baseName}-${take(suffix, 8)}'

resource logs 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: 'log-${baseName}'
  location: location
  properties: { sku: { name: 'PerGB2018' }, retentionInDays: 30 }
}

resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: 'appi-${baseName}'
  location: location
  kind: 'web'
  properties: { Application_Type: 'web', WorkspaceResourceId: logs.id }
}

resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: storageName
  location: location
  sku: { name: 'Standard_LRS' }
  kind: 'StorageV2'
  properties: {
    allowBlobPublicAccess: false
    minimumTlsVersion: 'TLS1_2'
    supportsHttpsTrafficOnly: true
  }
}

resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: kvName
  location: location
  properties: {
    tenantId: subscription().tenantId
    sku: { family: 'A', name: 'standard' }
    accessPolicies: []
  }
}

resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: acrName
  location: location
  sku: { name: 'Basic' }
  properties: { adminUserEnabled: false } // identities only, never admin passwords
}

resource mlWorkspace 'Microsoft.MachineLearningServices/workspaces@2024-04-01' = {
  name: 'mlw-${baseName}'
  location: location
  identity: { type: 'SystemAssigned' }
  properties: {
    friendlyName: baseName
    storageAccount: storage.id
    keyVault: keyVault.id
    applicationInsights: appInsights.id
    containerRegistry: acr.id // AML environment builds land in our ACR
    publicNetworkAccess: 'Enabled'
  }
}

resource acaEnv 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: 'cae-${baseName}'
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logs.properties.customerId
        sharedKey: logs.listKeys().primarySharedKey
      }
    }
  }
}

// First deployment creates the app with a public placeholder image so its identity + AcrPull exist
// before the first real image is pushed. CI redeploys app.bicep directly with the real image.
module api 'app.bicep' = {
  name: 'api'
  params: { baseName: baseName, location: location, acrName: acr.name, acaEnvName: acaEnv.name }
}

output acrName string = acr.name
output acrLoginServer string = acr.properties.loginServer
output storageAccountName string = storage.name
output mlWorkspaceName string = mlWorkspace.name
output acaEnvName string = acaEnv.name
output apiFqdn string = api.outputs.fqdn
