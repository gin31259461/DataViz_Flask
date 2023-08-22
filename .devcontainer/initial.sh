# Install mssql odbc 17 driver
DEBIAN_VERSION=11
curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
echo "deb [arch=amd64] https://packages.microsoft.com/debian/$DEBIAN_VERSION/prod bullseye main" | tee /etc/apt/sources.list.d/mssql-release.list
apt update
apt-get install -y msodbcsql17

apt list --upgradable
apt-get install -y libexpat1-dev
apt-get install -y libpango1.0-dev
