/* Determine CPU support for SIMD
 * Copyright 2004 Phil Karn, KA9Q
 */
#include <stdio.h>
#include "fec.h"

/* Various SIMD instruction set names */
char *Cpu_modes[] = {"Unknown","Portable C","x86 Multi Media Extensions (MMX)",
		   "x86 Streaming SIMD Extensions (SSE)",
		   "x86 Streaming SIMD Extensions 2 (SSE2)",
		   "PowerPC G4/G5 Altivec/Velocity Engine",
		   "ARM with NEON SIMD Instructions"};

enum cpu_mode Cpu_mode;

void find_cpu_mode(void){

  int f;
  if(Cpu_mode != UNKNOWN)
    return;

  // TODO: check if ARM arch really supports NEON
  Cpu_mode = ARM_NEON;
  fprintf(stderr,"SIMD CPU detect: %s\n",Cpu_modes[Cpu_mode]);
}
