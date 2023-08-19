# Install mssql odbc 17 driver
UBUNTU_VERSION=22.04
curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
echo "deb [arch=amd64] https://packages.microsoft.com/ubuntu/$UBUNTU_VERSION/prod jammy main" | tee /etc/apt/sources.list.d/mssql-release.list
apt update
apt-get install -y msodbcsql17

apt list --upgradable
apt-get install -y libexpat1-dev
apt-get install -y libpango1.0-dev
