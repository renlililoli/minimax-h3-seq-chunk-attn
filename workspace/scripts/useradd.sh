groupadd -g 1102 app
useradd -u 1091 -g 1102 -m -s /bin/bash app
su app