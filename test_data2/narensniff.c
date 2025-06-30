 /*============================================================================  
   =   		Passive sniffer by A.Narendra MCA JNT University, 2007.           =
   =																		  =
   =	 			Using PACKET SOCKETS ,ICMP,IP,UDP,TCP,ARP 		          =
   ============================================================================*/


#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <features.h>    /* for the glibc version number 2.1 only these headers will work */
#if __GLIBC__ >= 2 && __GLIBC_MINOR >= 1
#include <netpacket/packet.h>
#include <net/ethernet.h>     /* the L2 protocols */
#else
#include <asm/types.h>
#include <linux/if_packet.h>
#include <linux/if_ether.h>   /* The L2 protocols */
#endif
#include <sys/ioctl.h>
#include <linux/in.h>
#include <linux/if.h>
#include <linux/ip.h>
#include <linux/tcp.h>
#include <linux/udp.h>
#include <linux/icmp.h>
#include <stdlib.h>
#define DEVICE "eth0"
#define IP_DF 0x4000
#define IP_MF 0x2000

struct arphdr  {
  unsigned short ar_hrd;		/* format of hardware address	*/
  unsigned short ar_pro;		/* format of protocol address	*/
  unsigned char	 ar_hln;		/* length of hardware address	*/
  unsigned char	 ar_pln;		/* length of protocol address	*/
  unsigned short ar_op;			/* ARP opcode (command)		*/
  unsigned char	 ar_sha[ETH_ALEN];	/* sender hardware address	*/
  unsigned char	 ar_sip[4];		/* sender IP address		*/
  unsigned char	 ar_tha[ETH_ALEN];	/* target hardware address	*/
  unsigned char	 ar_tip[4];		/* target IP address		*/
};
  
/*----------------------------------------------------------------*/
/* A function to output the header fields of the received packets */
/*----------------------------------------------------------------*/
PrintHeadres(void *data)
{
  struct ethhdr eth;
  struct iphdr *ip;
  struct arphdr *arp;
  struct tcphdr *tcp;
  struct udphdr *udp;
  struct icmphdr *icmp;

  memcpy ((char *) &eth, data, sizeof(struct ethhdr));
  
  printf("==ETHERNET_HEADER============================\n");
  printf("MAC Source            ");
  printf(":%.2x:%.2x:%.2x:%.2x:%.2x:%.2x\n",
                eth.h_source[0], eth.h_source[1], eth.h_source[2],
	        eth.h_source[3], eth.h_source[4], eth.h_source[5]);

  printf("MAC Destination       ");
  printf(":%.2x:%.2x:%.2x:%.2x:%.2x:%.2x\n",
		eth.h_dest[0], eth.h_dest[1], eth.h_dest[2],
		eth.h_dest[3], eth.h_dest[4], eth.h_dest[5]);
  printf("Packet type ID field  :%#x\n", ntohs(eth.h_proto));

  if ((ntohs(eth.h_proto) == ETH_P_ARP) || 
      (ntohs(eth.h_proto) == ETH_P_RARP)) {
    arp = (struct arphdr *)(data + sizeof(struct ethhdr));

    printf("==ARP_HEADER=================================\n");
    printf("Format of hardware address     :%d\n", htons(arp->ar_hrd));
    printf("Format of protocol address     :%d\n", arp->ar_pro);
    printf("Length MAC                     :%d\n", arp->ar_hln);
    printf("Length IP                      :%d\n", arp->ar_pln);
    printf("ARP opcode                     :%d\n", htons(arp->ar_op));
    printf("Sender hardware address        :%.2x:%.2x:%.2x:%.2x:%.2x:%.2x\n",
						       arp->ar_sha[0],
						       arp->ar_sha[1],
						       arp->ar_sha[2],
						       arp->ar_sha[3],
						       arp->ar_sha[4],
						       arp->ar_sha[5],
						       arp->ar_sha[6]);
    printf("Sender IP address              :%d.%d.%d.%d\n",
    						       arp->ar_sip[0],
						       arp->ar_sip[1],
						       arp->ar_sip[2],
						       arp->ar_sip[3]);
    printf("Target hardware address        :%.2x:%.2x:%.2x:%.2x:%.2x:%.2x\n",
						       arp->ar_tha[0],
						       arp->ar_tha[1],
						       arp->ar_tha[2],
						       arp->ar_tha[3],
						       arp->ar_tha[4],
						       arp->ar_tha[5],
						       arp->ar_tha[6]);
    printf("Target IP address              :%d.%d.%d.%d\n",
						       arp->ar_tip[0],
						       arp->ar_tip[1],
						       arp->ar_tip[2],
						       arp->ar_tip[3]);  
      printf("#############################################\n");     
    }

    if (ntohs(eth.h_proto) == ETH_P_IP)
    {
      ip = (struct iphdr *)(data + sizeof(struct ethhdr));
       printf("==IP_HEADER==================================\n");
      printf("IP version            :%d\n", ip->version);
      printf("IP header length      :%d\n", ip->ihl);
      printf("TOS                   :%d\n", ip->tos);
      printf("Total length          :%d\n", ntohs(ip->tot_len));
      printf("ID                    :%d\n", ntohs(ip->id));
      printf("Fragment offset       :%#x\n", ntohs(ip->frag_off));
      printf("MF                    :%d\n", ntohs(ip->frag_off)&IP_MF?1:0);
      printf("DF                    :%d\n", ntohs(ip->frag_off)&IP_DF?1:0);
      printf("TTL                   :%d\n", ip->ttl);
      printf("Protocol              :%d\n", ip->protocol);
      printf("IP source             :%s\n", inet_ntoa(ip->saddr));
      printf("IP destination        :%s\n", inet_ntoa(ip->daddr));
	
      if ((ip->protocol) == IPPROTO_TCP) {
        tcp = (struct tcphdr *)(data + sizeof(struct ethhdr) + sizeof(struct iphdr));
        printf("==TCP_HEADER=================================\n");
        printf("Port source         :%d\n", ntohs(tcp->source));
        printf("Port destination    :%d\n", ntohs(tcp->dest));
        printf("Sequence number     :%d\n", ntohs(tcp->seq));
        printf("Ack number          :%d\n", ntohs(tcp->ack_seq));
        printf("Data offset         :%d\n", tcp->doff);
        printf("FIN:%d,", tcp->fin);
        printf("SYN:%d,", tcp->syn);
        printf("RST:%d,", tcp->rst);
        printf("PSH:%d,", tcp->psh);
        printf("ACK:%d,", tcp->ack);
        printf("URG:%d,", tcp->urg);
        printf("ECE:%d,", tcp->ece);
        printf("CWR:%d\n", tcp->cwr);
        printf("Window              :%d\n", ntohs(tcp->window));
	printf("Urgent pointer      :%d\n", tcp->urg_ptr);
      }

      if ((ip->protocol) == IPPROTO_UDP) {	
        udp = (struct udphdr *)(data + sizeof(struct ethhdr) + sizeof(struct iphdr));
        printf("==UDP_HEADER=================================\n");
        printf("Port source        :%d\n", ntohs(udp->source));
        printf("Port destination   :%d\n", ntohs(udp->dest));
        printf("Length             :%d\n", ntohs(udp->len));
      }

      if ((ip->protocol) == IPPROTO_ICMP) {	
        icmp = (struct icmphdr *)(data + sizeof(struct ethhdr) + sizeof(struct iphdr));
	printf("==ICMP_HEADER================================\n");
        printf("Type        :%d\n", icmp->type);
        printf("Code        :%d\n", icmp->code);
      }

      printf("#############################################\n");
    }
}



struct IPPacket {
    unsigned ACKFLAG : 3
    unsigned RSTFLAG : 4
};

IPPacket pkt;
pkt.ACKFLAG = 2




/*-------------------------------------------------------------*/
/* A function to output the received data as a hex&ascii dump. */
/*-------------------------------------------------------------*/
void Dump(void* data, int len)
{  
  unsigned char *buf = data;
  int i;
  int poz = 0;
  char str[17];
  memset(str, 0, 17);
  for (i = 0; i < len; i++)
  {
    if (poz % 16 == 0)  {
      printf("  %s\n%04X: ", str, poz);
      memset(str, 0, 17);
    }
    
    if (buf[poz] < ' ' || buf[poz] >= 127)
      str[poz%16] = '.';
    else
      str[poz%16] = buf[poz];
      
    printf("%02X ", buf[poz++]);
  }
  printf("  %*s\n\n", 16 + (16 - len % 16) * 2, str);
}

/*---------------------*/
/* The main() function */
/*---------------------*/
int main(int argc, char* argv[])
{
  int sd;
  int n = 0;
  int packet = 0;
  struct ifreq ifr;
  char buf[1500];

  fprintf(stderr, "===================================================\n");  
  fprintf(stderr, "= Simple passive sniffer by Narendra JNTU , 2007. =\n");
  fprintf(stderr, "=    [-d] - dump a block of data in hex&ascii.    =\n");
  fprintf(stderr, "===================================================\n");

  if ( (sd = socket(PF_PACKET, SOCK_RAW, htons(ETH_P_ALL))) < 0) {
    perror("socket() failed");
    exit(-1);
  }

  /* Switching the interface into the promiscuous mode */
  strcpy(ifr.ifr_name, DEVICE);
  if (ioctl(sd, SIOCGIFFLAGS, &ifr) < 0) {
    perror("ioctl() failed");
    close(sd);
    exit(-1);
  }
	
  ifr.ifr_flags |= IFF_PROMISC;

  if (ioctl(sd, SIOCSIFFLAGS, &ifr) < 0) {
    perror("ioctl() failed");
    close(sd);
    exit(-1);
  }

  /* Receiving packets in an endless loop */
  while (1)
  {
    n = recvfrom(sd, buf, sizeof(buf), 0, 0, 0);
    printf("#############################################\n");
    printf("Packet #%d (%d bytes read)\n", ++packet, n);
    
    PrintHeadres(buf); /* Outputting the header fields of the received packets. */
    
    /* If the -d parameter was specified in the command line, showing the received data as a hex&ascii dump */
    
    if (argc == 2) {
      if (!strcmp(argv[1], "-d"))
        Dump(buf, n);
    }
  
    printf("\n");  
   }

  return 0;
}
 
