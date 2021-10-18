# This is an auto generated Dockerfile for ros:desktop-full
# generated from docker_images/create_ros_image.Dockerfile.em
FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive 

# Requirements
# BEGIN INSTALL packages
RUN apt-get update

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates build-essential dbus-tests \ 
        mesa-utils nano python3-pip dirmngr gnupg2 lsb-release
        
RUN apt-get update && apt-get upgrade -y

#END INSTALL PACKAGES
RUN rm -rf /var/lib/apt/lists/*


# Correction warning node turtle: process 3135: D-Bus library appears to be incorrectly set up; 
# failed to read machine uuid: UUID file '/etc/machine-id' should contain a hex string of length 32, not length 0, with no other text
# See the manual page for dbus-uuidgen to correct this issue.
RUN mkdir -p /var/lib/dbus/
RUN touch /var/lib/dbus/machine-id
RUN dbus-uuidgen > /var/lib/dbus/machine-id

# Add user
RUN adduser --quiet --disabled-password qtuser && usermod -a -G audio qtuser

CMD ["/bin/bash"]