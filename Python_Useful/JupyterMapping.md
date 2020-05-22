# Problem

You want to use Jupyter remotely.  X11 forwarding is too slow for this.

# Solution

SSH port forwarding!

# The recipe

## On your server

~~~{.sh}
cd directory
#Map a jupyter process to port 8889 on server
jupyter notebook --no-browser --port=8889
~~~

**Note:** Port number is semi-arbitrary.  Just avoid any existing services! :)

## On your client

~~~{.sh}
#Map port 8888 on your client to 8889 on the remote host:
ssh -N -f -L localhost:8888:localhost:8889 $LINUXBOX
~~~

Then, navigate your browser to URL localhost:8888