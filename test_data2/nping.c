 /*======================================================================  
   = 	   Custom ping tool by A.Narendra MCA JNT University, 2007.     =
   =																	=
   =	 	           Using  Raw scokets,ICMP						    =
   ======================================================================*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/ip.h>
#include <netinet/ip_icmp.h>
#include <netdb.h>
#include <sys/time.h>
#include <signal.h>
#include <unistd.h>

#define BUFSIZE 1500

int sd;                      /* Socket descriptor */
pid_t pid;                   /* Program's PID */
struct sockaddr_in servaddr; /* Structure for sending a packet */
struct sockaddr_in from;     /* Structure for receiving a packet */

double tmin = 999999999.0;   /* Minimum round-trip time */
double tmax = 0;             /* Maximum round-trip time */
double tsum = 0;             /* Sum of all times for calculating the
                                average time */
int nsent = 0;               /* Number of sent packets */
int nreceived = 0;           /* Number of received packets */

/* Function prototypes */
void pinger(void);
void output(char *, int, struct timeval *);
void catcher(int);
void tv_sub(struct timeval *, struct timeval *);
unsigned short in_cksum(unsigned short *, int);

/*---------------------*/
/* The main() function */
/*---------------------*/
int main(int argc, char *argv[])
{
  int size;
  int fromlen;
  int n;
  struct timeval tval;
  char recvbuf[BUFSIZE];
  struct hostent *hp;
  struct sigaction act;
  struct itimerval timer;
  const int on = 1;

  if (argc != 2) {
    fprintf(stderr, "Usage: %s <hostname>\n", argv[0]);
    exit(-1);
  }

  pid = getpid();

  /* Setting the handler for the SIGALRM and SIGINT signals */
  memset(&act, 0, sizeof(act));
  /* Assigning the catcher() function as the handler */
  act.sa_handler = &catcher;
  sigaction(SIGALRM, &act, NULL);
  sigaction(SIGINT, &act, NULL);

  if ( (hp = gethostbyname(argv[1])) == NULL) {
    herror("gethostbyname() failed");
    exit(-1);
  }

  if ( (sd = socket(PF_INET, SOCK_RAW, IPPROTO_ICMP)) < 0) {
    perror("socket() failed");
    exit(-1);
  }

  /* Restoring the initial rights */
  setuid(getuid());
  /* Enabling the broadcasting capability */
  setsockopt(sd, SOL_SOCKET, SO_BROADCAST, &on, sizeof(on));
  /* Increasing the receiving buffer size */
  size = 60 *1024;
  setsockopt(sd, SOL_SOCKET, SO_RCVBUF, &size, sizeof(size));

  /* Starting a timer to send the SIGALRM signal */
  /* Timer kicks in after 1 microsecond */
  timer.it_value.tv_sec=0;
  timer.it_value.tv_usec=1;
  /* Timer fires every second */
  timer.it_interval.tv_sec=1;
  timer.it_interval.tv_usec=0;
  /* Starting the real-time timer */
  setitimer(ITIMER_REAL, &timer, NULL);

  bzero(&servaddr, sizeof(servaddr));
  servaddr.sin_family = AF_INET;
  servaddr.sin_addr= *((struct in_addr *) hp->h_addr);

  fromlen = sizeof(from);

  /* Starting an endless loop to receive packets */
  while (1) {
    n = recvfrom(sd, recvbuf, sizeof(recvbuf), 0,
               (struct sockaddr *)&from, &fromlen);

    if (n < 0) {
      if (errno == EINTR)
        continue;
      perror("recvfrom() failed");
      continue;
    }

    /* Determining the current system time */
    gettimeofday(&tval, NULL);

    /* Calling the function to parse the received */
    /* packet and display the data */
    output(recvbuf, n, &tval);
  }

  return 0;
}

/*--------------------------------------------*/
/* Parsing the packet and displaying the data */
/*--------------------------------------------*/
void output(char *ptr, int len, struct timeval *tvrecv)
{
  int iplen;
  int icmplen;
  struct ip *ip;
  struct icmp *icmp;
  struct timeval *tvsend;
  double rtt;

  ip = (struct ip *) ptr; /* Start of the IP header */
  iplen = ip->ip_hl << 2; /* Length of the IP header */

  icmp = (struct icmp *) (ptr + iplen); /* Start of the ICMP header */
  if ( (icmplen = len - iplen) < 8)     /* Length of the ICMP header */
    fprintf(stderr, "icmplen (%d) < 8", icmplen);

  if (icmp->icmp_type == ICMP_ECHOREPLY) {

    if (icmp->icmp_id != pid)
      return; /* Reply is to another ping's echo request */

    tvsend = (struct timeval *) icmp->icmp_data;
    tv_sub(tvrecv, tvsend);

    /* Round-trip time */
    rtt = tvrecv->tv_sec * 1000.0 + tvrecv->tv_usec / 1000.0;

    nreceived++;

    tsum += rtt;
    if (rtt < tmin)
      tmin = rtt;
    if (rtt > tmax)
      tmax = rtt;

    printf("%d bytes from %s: icmp_seq=%u, ttl=%d, time=%.3f ms\n",
           icmplen, inet_ntoa(from.sin_addr),

           icmp->icmp_seq, ip->ip_ttl, rtt);
  }

}

/*-------------------------------------------------*/
/* Forming and sending an ICMP echo request packet */
/*-------------------------------------------------*/
void pinger(void)
{
  int icmplen;
  struct icmp *icmp;
  char sendbuf[BUFSIZE];

  icmp = (struct icmp *) sendbuf;

  /* Filling all fields of the ICMP message */
  icmp->icmp_type = ICMP_ECHO;
  icmp->icmp_code = 0;
  icmp->icmp_id = pid;
  icmp->icmp_seq = nsent++;
  gettimeofday((struct timeval *) icmp->icmp_data, NULL);

  /* Length is 8 bytes of ICMP header and 56 bytes of data */
  icmplen = 8 + 56;
  /* Checksum for the ICMP header and data */
  icmp->icmp_cksum = 0;
  icmp->icmp_cksum = in_cksum((unsigned short *) icmp, icmplen);

  if (sendto(sd, sendbuf, icmplen, 0,
    (struct sockaddr *)&servaddr, sizeof(servaddr)) < 0) {
    perror("sendto() failed");
    exit(-1);
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

/*------------------------------------------------*/
/* The handler for the SIGALRM and SIGINT signals */
/*------------------------------------------------*/

void catcher(int signum)
{
  if (signum == SIGALRM)
  {
    pinger();
     return;
  } else if (signum == SIGINT) {
    printf("\n--- %s ping statistics ---\n", inet_ntoa(servaddr.sin_addr));
    printf("%d packets transmitted, ", nsent);
    printf("%d packets received, ", nreceived);
    if (nsent)
    {
      if (nreceived > nsent)
       printf("-- somebody's printing packets!");
      else
       printf("%d%% packet loss",
                      (int) (((nsent-nreceived)*100) /
                     nsent));
    }
    printf("\n");
    if (nreceived)
      printf("round-trip min/avg/max = %.3f/%.3f/%.3f ms\n",
              tmin,
              tsum / nreceived,
              tmax);
    fflush(stdout);
    exit(-1);
  }
}

/*--------------------------*/
/* Calculating the checksum */
/*--------------------------*/
unsigned short in_cksum(unsigned short *addr, int len)
{
  unsigned short result;
  unsigned int sum = 0;

  /* Adding all 2-byte words */
  while (len > 1) {
    sum += *addr++;
    len -= 2;
  }

  /* If there is a byte left over, adding it to the sum */
  if (len == 1)
    sum += *(unsigned char*) addr;

  sum = (sum >> 16) + (sum & 0xFFFF); /* Adding the carry */
  sum += (sum >> 16);                 /* Adding the carry again */
  result = ~sum;                      /* Inverting the result */
  return result;
}

