 /*======================================================================  
   = 	 Custom tracerout tool by A.Narendra MCA JNT University, 2007.  =
   =																	=
   =	 	           Using  Raw scokets,ICMP							=
   ======================================================================*/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <netinet/ip.h>
#include <netinet/ip_icmp.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <sys/time.h>
#include <unistd.h>

#define BUFSIZE 1500

int sd;   	/* Socket descriptor */
pid_t pid;	/* PID of "our" process */
struct sockaddr_in sasend; /* The sockaddr() structure for sending a packet. */
struct sockaddr_in sarecv; /* The sockaddr() structure for receiving a packet */
struct sockaddr_in salast; /* The last sockaddr() structure for receiving a packet */

int ttl;
int probe;
int max_ttl = 30;	/* Maximum value for the TTL field. */
int nprobes = 3;  /* Number of probing packets */

/* Function prototypes */
int output(int, struct timeval *);
void tv_sub(struct timeval *, struct timeval *);
unsigned short in_cksum(unsigned short *, int);

/*------------------------*/
/* The main() function    */
/*------------------------*/
int main(int argc, char *argv[])
{
  int seq;
  int code;
  int done;
  double rtt;
  struct timeval *tvsend;
  struct timeval tvrecv;
  struct hostent *hp;  
  int icmplen;
  struct icmp *icmp;
  char sendbuf[BUFSIZE];

  if (argc != 2) {
    fprintf(stderr, "Usage: %s <hostname>\n", argv[0]);
    exit(-1);
  }

  pid = getpid();

  if ( (hp = gethostbyname(argv[1])) == NULL) {
    herror("gethostbyname() failed");
    exit(-1);
  }

  if ( (sd = socket(PF_INET, SOCK_RAW, IPPROTO_ICMP)) < 0) {
    perror ("socket() failed");
    exit(-1);
  }
	
  setuid(getuid());

  bzero(&sasend, sizeof(sasend));
  sasend.sin_family = AF_INET;
  sasend.sin_addr= *((struct in_addr *) hp->h_addr);
 
  seq = 0;
  done = 0;
  for (ttl = 1; ttl <= max_ttl && done == 0; ttl++) {
    setsockopt(sd, SOL_IP, IP_TTL, &ttl, sizeof(int));
    bzero(&salast, sizeof(salast));

    printf("%2d  ", ttl);
    fflush(stdout);

    for (probe = 0; probe < nprobes; probe++) {

      icmp = (struct icmp *) sendbuf;
      
      /* Filling all fields of the ICMP message */
      icmp->icmp_type = ICMP_ECHO;
      icmp->icmp_code = 0;
      icmp->icmp_id = pid;
      icmp->icmp_seq = ++seq;
      tvsend = (struct timeval *) icmp->icmp_data;
      gettimeofday(tvsend, NULL);
      
      /* The length is 8 bytes of ICMP header and 56 bytes of data */
      icmplen = 8 + 56;
      /* The checksum for the ICMP header and the data */
      icmp->icmp_cksum = 0;
      icmp->icmp_cksum = in_cksum((unsigned short *) icmp, icmplen);

      if (sendto(sd, sendbuf, icmplen, 0, (struct sockaddr *)&sasend, sizeof(sasend)) < 0) {
        perror("sendto() failed");
        exit(-1);
      }
      
      if ( (code = output(seq, &tvrecv)) == -3)
        printf(" *");
      else {
        if (memcmp(&sarecv.sin_addr, &salast.sin_addr, sizeof(sarecv.sin_addr)) != 0) {
          if ( (hp = gethostbyaddr(&sarecv.sin_addr, sizeof(sarecv.sin_addr), sarecv.sin_family)) != 0)
            printf(" %s (%s)", inet_ntoa(sarecv.sin_addr), hp->h_name);
          else
            printf(" %s", inet_ntoa(sarecv.sin_addr));
          memcpy(&salast.sin_addr, &sarecv.sin_addr, sizeof(salast.sin_addr));
        }
	
        tv_sub(&tvrecv, tvsend);
        rtt = tvrecv.tv_sec * 1000.0 + tvrecv.tv_usec / 1000.0;
        printf("  %.3f ms", rtt);

        if (code == -1)
          ++done;
      }
      
      fflush(stdout);
    }
    
    printf("\n");
  }

  return 0;
}

/*---------------------------------------------------------------*/
/* Parsing a received packet                                     */
/*                                                               */
/* The function returns:                                         */
/* -3 when the wait time expires                                 */
/* -2 when an ICMP time exceeded in transit message is received; */
/*    the program continues executing.                           */
/* -1 when an ICMP Echo Reply message is received;               */
/*    the program terminates execution.                          */
/*---------------------------------------------------------------*/
int output(int seq, struct timeval *tv)
{ 
  int n;
  int len;
  int hlen1;
  int hlen2;
  struct ip *ip;
  struct ip *hip;
  struct icmp *icmp;
  struct icmp *hicmp;
  double rtt;
  char recvbuf[BUFSIZE];
  fd_set fds;
  struct timeval wait;

  wait.tv_sec = 4;  /* Waiting for a reply for 4 seconds, the longest. */
  wait.tv_usec = 0;

  for (;;) {
    len = sizeof(sarecv);

    FD_ZERO(&fds);
    FD_SET(sd, &fds);

    if (select(sd+1, &fds, NULL, NULL, &wait) > 0)
      n = recvfrom(sd, recvbuf, sizeof(recvbuf), 0, (struct sockaddr*)&sarecv, &len);
     else if (!FD_ISSET(sd, &fds))
      return (-3);
    else
      perror("recvfrom() failed");

    gettimeofday(tv, NULL);

    ip = (struct ip *) recvbuf;	/* Start of the IP header  */
    hlen1 = ip->ip_hl << 2;        /* Length of the IP header */
	
    icmp = (struct icmp *) (recvbuf + hlen1); /* Start of the ICMP header */

     if (icmp->icmp_type == ICMP_TIMXCEED && 
        icmp->icmp_code == ICMP_TIMXCEED_INTRANS) {
      hip = (struct ip *)(recvbuf + hlen1 + 8);
      hlen2 = hip->ip_hl << 2;
      hicmp = (struct icmp *) (recvbuf + hlen1 + 8 + hlen2);
      if (hicmp->icmp_id == pid && hicmp->icmp_seq == seq)  
        return (-2);
    }

    if (icmp->icmp_type == ICMP_ECHOREPLY &&
        icmp->icmp_id == pid &&
        icmp->icmp_seq == seq)
      return (-1);
  }

}
  
/*------------------------------------------------*/
/* Subtracting one timeval structure from another */
/*------------------------------------------------*/ 
void tv_sub(struct timeval *out, struct timeval *in)
{
  if ( (out->tv_usec -= in->tv_usec) < 0) {
    out->tv_sec--;
    out->tv_usec += 1000000;
  }
  out->tv_sec -= in->tv_sec;
}

/*------------------------------*/
/* Calculating the checksum     */
/*------------------------------*/
unsigned short in_cksum(unsigned short *addr, int len)
{
  unsigned short result;
  unsigned int sum = 0;

  /* Adding all two-byte words */    
  while (len > 1) {
    sum += *addr++;
    len -= 2;
  }
  
  /* If there is a byte is left over, adding it to the sum */
  if (len == 1)
    sum += *(unsigned char*) addr;
    
  sum = (sum >> 16) + (sum & 0xFFFF);  /* Adding the carry */
  sum += (sum >> 16);			    /* Adding the carry again */
  result = ~sum;				    /* Inverting the result */
  return result;
}
 
