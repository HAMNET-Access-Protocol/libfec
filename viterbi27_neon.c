/* K=7 r=1/2 Viterbi decoder optimized for arm neon
 * Based on the SSE2 decoder by Phil Karn, KA9Q
 * May be used under the terms of the GNU Lesser General Public License (LGPL)
 */
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <limits.h>
#include <arm_neon.h>
#include "fec.h"

typedef union { unsigned char c[64]; int8x16_t v[4]; } metric_t;
typedef union { unsigned int w[2]; unsigned char c[8]; unsigned short s[4]; uint8x8_t v[1];} decision_t;
union branchtab27 { unsigned char c[32]; uint8x16_t v[2];} Branchtab27_neon[2];
static int Init = 0;

/* State info for instance of Viterbi decoder
 * Don't change this without also changing references in [mmx|sse|sse2]bfly29.s!
 */
struct v27 {
  metric_t metrics1; /* path metric buffer 1 */
  metric_t metrics2; /* path metric buffer 2 */
  decision_t *dp;          /* Pointer to current decision */
  metric_t *old_metrics,*new_metrics; /* Pointers to path metrics, swapped on every bit */
  decision_t *decisions;   /* Beginning of decisions for block */
};

/* Initialize Viterbi decoder for start of new frame */
int init_viterbi27_neon(void *p,int starting_state){
  struct v27 *vp = p;
  int i;

  if(p == NULL)
	return -1;
  for(i=0;i<64;i++)
	vp->metrics1.c[i] = 63;

  vp->old_metrics = &vp->metrics1;
  vp->new_metrics = &vp->metrics2;
  vp->dp = vp->decisions;
  vp->old_metrics->c[starting_state & 63] = 0; /* Bias known start state */
  return 0;
}

void set_viterbi27_polynomial_neon(int polys[2]){
  int state;

  for(state=0;state < 32;state++){
    Branchtab27_neon[0].c[state] = (polys[0] < 0) ^ parity((2*state) & abs(polys[0])) ? 255 : 0;
    Branchtab27_neon[1].c[state] = (polys[1] < 0) ^ parity((2*state) & abs(polys[1])) ? 255 : 0;
  }
  Init++;
}

/* Create a new instance of a Viterbi decoder */
void *create_viterbi27_neon(int len){
  struct v27 *vp;

  if(!Init){
    int polys[2] = { V27POLYA, V27POLYB };
    set_viterbi27_polynomial_neon(polys);
  }
  if((vp = malloc(sizeof(struct v27))) == NULL)
     return NULL;
  if((vp->decisions = malloc((len+6)*sizeof(decision_t))) == NULL){
    free(vp);
    return NULL;
  }
  init_viterbi27_neon(vp,0);

  return vp;
}

/* Viterbi chainback */
int chainback_viterbi27_neon(
      void *p,
      unsigned char *data, /* Decoded output data */
      unsigned int nbits, /* Number of data bits */
      unsigned int endstate){ /* Terminal encoder state */
  struct v27 *vp = p;
  decision_t *d;

  if(p == NULL)
    return -1;
  d = vp->decisions;
  /* Make room beyond the end of the encoder register so we can
   * accumulate a full byte of decoded data
   */
  endstate %= 64;
  endstate <<= 2;

  /* The store into data[] only needs to be done every 8 bits.
   * But this avoids a conditional branch, and the writes will
   * combine in the cache anyway
   */
  d += 6; /* Look past tail */
  while(nbits-- != 0){
    int k;

    k = (d[nbits].w[(endstate>>2)/32] >> ((endstate>>2)%32)) & 1;
    data[nbits>>3] = endstate = (endstate >> 1) | (k << 7);
  }
  return 0;
}

/* Delete instance of a Viterbi decoder */
void delete_viterbi27_neon(void *p){
  struct v27 *vp = p;

  if(vp != NULL){
    free(vp->decisions);
    free(vp);
  }
}

// NEON equivalent for _mm_movemask_epi8
// Taken from: https://stackoverflow.com/questions/11870910/sse-mm-movemask-epi8-equivalent-method-for-arm-neon
int vmovmaskq_u8(uint8x16_t input)
{
    // Example input (half scale):
    // 0x89 FF 1D C0 00 10 99 33

    // Shift out everything but the sign bits
    // 0x01 01 00 01 00 00 01 00
    uint16x8_t high_bits = vreinterpretq_u16_u8(vshrq_n_u8(input, 7));

    // Merge the even lanes together with vsra. The '??' bytes are garbage.
    // vsri could also be used, but it is slightly slower on aarch64.
    // 0x??03 ??02 ??00 ??01
    uint32x4_t paired16 = vreinterpretq_u32_u16(
                              vsraq_n_u16(high_bits, high_bits, 7));
    // Repeat with wider lanes.
    // 0x??????0B ??????04
    uint64x2_t paired32 = vreinterpretq_u64_u32(
                              vsraq_n_u32(paired16, paired16, 14));
    // 0x??????????????4B
    uint8x16_t paired64 = vreinterpretq_u8_u64(
                              vsraq_n_u64(paired32, paired32, 28));
    // Extract the low 8 bits from each lane and join.
    // 0x4B
    return vgetq_lane_u8(paired64, 0) | ((int)vgetq_lane_u8(paired64, 8) << 8);
}

/* Update decoder with a block of demodulated symbols
 * Note that nbits is the number of decoded data bits, not the number
 * of symbols!
 */
int update_viterbi27_blk_neon(void *p,unsigned char *syms,int nbits){
  struct v27 *vp = p;
  decision_t *d;

  if(p == NULL)
    return -1;
  d = (decision_t *)vp->dp;
  while(nbits--){
    uint8x16_t sym0v,sym1v;
    void *tmp;
    int i;

    /* Splat the 0th symbol across sym0v, the 1st symbol across sym1v, etc */
    sym0v = vdupq_n_u8(syms[0]);
    sym1v = vdupq_n_u8(syms[1]);
    syms += 2;
    for(i=0;i<2;i++){
      int8x16_t metric,m_metric,m0,m1,m2,m3,survivor0,survivor1;
      uint8x16_t decision0, decision1;
      /* Form branch metrics */
      metric = vreinterpretq_s8_u8(vrhaddq_u8(veorq_u8(Branchtab27_neon[0].v[i],sym0v),
    		  	  	  	  	  	  	  	       veorq_u8(Branchtab27_neon[1].v[i],sym1v)));
      /* There's no packed bytes right shift in SSE2, so we use the word version and mask
       * (I'm *really* starting to like Altivec...)
       * NEON NOTE: - ARM supports this with vshrq_n_u8
       *            - we switched to 4bit branch metrics, because 5bit metric code
       *              does not work good in high Eb/N0 environments
       */
      metric = vreinterpretq_s8_u8(vshrq_n_u8((uint8x16_t)metric,4));
      //metric = _mm_and_si128(metric,_mm_set1_epi8(31));
      m_metric = vsubq_s8(vdupq_n_s8(15),metric);
      /* Add branch metrics to path metrics */
      m0 = vaddq_s8(vp->old_metrics->v[i],metric);
      m3 = vaddq_s8(vp->old_metrics->v[2+i],metric);
      m1 = vaddq_s8(vp->old_metrics->v[2+i],m_metric);
      m2 = vaddq_s8(vp->old_metrics->v[i],m_metric);

      /* Compare and select, using modulo arithmetic */
      decision0 = vcgtq_s8(vsubq_s8(m0,m1),vdupq_n_s8(0));
      decision1 = vcgtq_s8(vsubq_s8(m2,m3),vdupq_n_s8(0));
      survivor0 = vorrq_s8(vandq_s8((int8x16_t)decision0,m1),vandq_s8(vmvnq_s8((int8x16_t)decision0),m0));
      survivor1 = vorrq_s8(vandq_s8((int8x16_t)decision1,m3),vandq_s8(vmvnq_s8((int8x16_t)decision1),m2));
      /* Pack each set of decisions into 16 bits */
      // ARM NEON: there is no movemask intrinsic for NEON. had to create one
      uint8x16x2_t zip = vzipq_u8(decision0,decision1);
      d->s[2*i] = vmovmaskq_u8(zip.val[0]);
      d->s[2*i+1] = vmovmaskq_u8(zip.val[1]);

      /* Store surviving metrics */
      int8x16x2_t zip2 = vzipq_s8(survivor0,survivor1);
      vp->new_metrics->v[2*i] = zip2.val[0];
      vp->new_metrics->v[2*i+1] = zip2.val[1];
    }
    d++;
    /* Swap pointers to old and new metrics */
    tmp = vp->old_metrics;
    vp->old_metrics = vp->new_metrics;
    vp->new_metrics = tmp;
  }
  vp->dp = d;
  return 0;
}

