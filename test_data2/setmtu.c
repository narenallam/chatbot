#include <stdio.h>
#include <string.h>

#include <linux/capability.h>

#include <sys/ioctl.h>
#include <sys/socket.h>

#include <net/if.h>
#include <net/if_packet.h>
#include <netpacket/packet.h>

#include <net/ethernet.h>

int main()
{
	struct ifreq ifr;
	int s;
	char *dev = "eth0";//Can be eth0,eth1...etc
        int mtu;
        printf("Enter MTU :");
	scanf("%d",&mtu);

	s = socket(AF_INET,SOCK_DGRAM,0);
	if (s < 0)
		printf("Error: Socket creation\n");

	memset(&ifr, 0, sizeof(ifr));
	strncpy(ifr.ifr_name, dev, sizeof(ifr.ifr_name));
	ifr.ifr_mtu = mtu;
	if (ioctl(s, SIOCSIFMTU, &ifr) < 0) {
		printf("Error:SIOCSIFMTU\n");
		close(s);
		return -1;
	}
	close(s);
return 0;
}
