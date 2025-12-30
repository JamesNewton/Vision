# Vision
Just some experiments with vision 

## Setup
An Aaeon Boxer B with an extra many GB SDcard to provide additional storage. These little units are available for next to nothing today, used or "new old stock". They have:
- 4 GB RAM, 2GB Swap (which removes most of the memory limits that plague stock Nanos)
- 16 GB of internal eMMC (which is almost completely used with the stock OS install)
- Quad core ARM A57 Processor
- NVIDIA Jetson nano GPU, etc...

Because there is so little space available on the internal drive, docker needs to use the sdcard
and it needs to automount. Make a perminant mount point:
<br>`sudo mkdir -p /mnt/sdcard`
<br> use 
<br>`sudo blkid`
<br> to find the UUID of the sdcard
<br>`sudo nano /etc/fstab`
<br> and add
<br>`UUID=<sdcard UUID>  /mnt/sdcard  ext4  defaults  0  2`
<br> then save that and restart or continue with
<br>`sudo mount -a`
<br>`sudo nano /etc/docker/daemon.json`
<br> add `"data-root": "/mnt/sdcard/docker-data"`
<br> and restart with
<br>`sudo systemctl restart docker`

 I also backed up the boxer OS just incase it got fried:
<br>`sudo dd if=/dev/mmcblk0 of=/media/your_user/USB_NAME/boxer_backup.img bs=4M status=progress`

<br> The dustynv docker container is the basis for all the installed libraries.
<br>`sudo docker pull dustynv/jetson-inference:r32.4.3`

To resolve an error with mobilenet not being found, we download it manually 
```
mkdir -p /mnt/sdcard/networks
cd /mnt/sdcard/networks
wget https://github.com/dusty-nv/jetson-inference/releases/download/model-mirror-190618/SSD-Mobilenet-v2.tar.gz
tar -zxvf SSD-Mobilenet-v2.tar.gz
```
Run with:
```
sudo docker run --runtime nvidia -it --rm \
    --network host \
    -e DISPLAY=:0 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /mnt/sdcard:/mnt/sdcard \
    dustynv/jetson-inference:r32.4.3

export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
cd /mnt/sdcard
python3 <whicever>.py
```

## Code
The focus of the code is to run efficiently with low energy use by NOT just running compute intensive things like image recognition on every frame, and by reducing the frame rate to just a few per second. Instead, it uses simple OpenCV methods for frame subtraction to detect motion, and *then* do the fancy stuff when there is a reason for it.

- `smart_sentry.py` basic code to read out an old standard video stream and process 1 frame in n from that. 
- `slow_sentry.py` switches to single image captures at a much slower rate, which still works just fine.
- `multi-sentry.py` supports multiple cameras, including more modern streaming only units like the topo, white still limiting processing. It also supports a tasmota device to ring a doorbell or flash a light when a person is seen. It also writes out a file to trigger a notification via a web server script.

