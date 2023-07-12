/******************************************************************************
 * Copyright (C) 2022 Maxim Integrated Products, Inc., All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL MAXIM INTEGRATED BE LIABLE FOR ANY CLAIM, DAMAGES
 * OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 * Except as contained in this notice, the name of Maxim Integrated
 * Products, Inc. shall not be used except as stated in the Maxim Integrated
 * Products, Inc. Branding Policy.
 *
 * The mere transfer of this software does not imply any licenses
 * of trade secrets, proprietary technology, copyrights, patents,
 * trademarks, maskwork rights, or any other form of intellectual
 * property whatsoever. Maxim Integrated Products, Inc. retains all
 * ownership rights.
 *
 ******************************************************************************/
#include <string.h>
#include <math.h>

#include "board.h"
#include "mxc_device.h"
#include "mxc_delay.h"
#include "mxc.h"
#include "utils.h"
#include "camera.h"
#include "faceID.h"
#include "facedetection.h"
#include "record.h"
#include "utils.h"
#include "MAXCAM_Debug.h"
#include "cnn_1.h"
#include "cnn_2.h"
#include "cnn_3.h"
#include "led.h"
#include "lp.h"
#include "uart.h"
#include "math.h"
#include "post_process.h"
#include "flc.h"
#include "tft_utils.h"
#include "faceID.h"
#include "facedetection.h"
#include "weights_3.h"
#include "baseaddr.h"


#define S_MODULE_NAME "record"


/***** Definitions *****/
#define TEST_ADDRESS (MXC_FLASH_MEM_BASE + MXC_FLASH_MEM_SIZE) - (1 * MXC_FLASH_PAGE_SIZE)

#define STATUS_ADDRESS (MXC_FLASH_MEM_BASE + MXC_FLASH_MEM_SIZE) - (2 * MXC_FLASH_PAGE_SIZE)
/*
    ^ Points to last page in flash, which is guaranteed to be unused by this small example.
    For larger applications it's recommended to reserve a dedicated flash region by creating
    a modified linkerfile.
*/
#define MAGIC 0xFEEDBEEF
#define TEST_VALUE 0xDEADBEEF

/*
Example Flash Database :
        MAGICKEY
        NNNNIIIL
        NNNNNNNN
        EEEEEEEE
        ........
        ........
        ........
        NNNNIIIL
        NNNNNNNN
        EEEEEEEE
        ........
        N : Name of the person (6 bytes)
        I : Unique ID of the person (12 bits)
        L : Embeddings count of the person (4 bits)
        E : Embeddings of the person (L * 64 bytes)

Example Status Field :
        MAGICKEY
        TTTTTTTT 
        T : Total number of people in the database (32 bits)
*/

/***** Globals *****/
volatile uint32_t isr_cnt;
volatile uint32_t isr_flags;
extern volatile int32_t output_buffer[16];
extern unsigned int touch_x, touch_y;
extern volatile uint8_t face_detected;
extern volatile uint8_t capture_key;
extern volatile uint8_t record_mode;
static const uint32_t kernels_3[] = KERNELS_3;
static const uint32_t baseaddr[] = BASEADDR;





void FLC0_IRQHandler(void)
{
    uint32_t temp;
    isr_cnt++;
    temp = MXC_FLC0->intr;

    if (temp & MXC_F_FLC_INTR_DONE) {
        MXC_FLC0->intr &= ~MXC_F_FLC_INTR_DONE;
        printf(" -> Interrupt! (Flash operation done)\n\n");
    }

    if (temp & MXC_F_FLC_INTR_AF) {
        MXC_FLC0->intr &= ~MXC_F_FLC_INTR_AF;
        printf(" -> Interrupt! (Flash access failure)\n\n");
    }

    isr_flags = temp;
}

//============================================================================
int init_db()
{

int err = 0;
 printf("Erasing page 64 of flash (addr 0x%x)...\n", TEST_ADDRESS);
            err = MXC_FLC_PageErase(TEST_ADDRESS);
            if (err) {
            printf("Failed with error code %i\n", TEST_ADDRESS, err);
            return err; }

            PR_DEBUG("Magic Value not matched, Initializing flash\n");
            err = MXC_FLC_Write32(TEST_ADDRESS, MAGIC);

            if (err) {
            printf("Failed to write magic value to 0x%x with error code %i!\n", TEST_ADDRESS, err);
            return err;}

 printf("Erasing page 63 of flash (addr 0x%x)...\n", STATUS_ADDRESS);
            err = MXC_FLC_PageErase(STATUS_ADDRESS);
            if (err) {
            printf("Failed with error code %i\n", STATUS_ADDRESS, err);
            return err; }

            PR_DEBUG("Magic Value not matched, Initializing flash\n");
            err = MXC_FLC_Write32(STATUS_ADDRESS, MAGIC);

            if (err) {
            printf("Failed to write magic value to 0x%x with error code %i!\n", STATUS_ADDRESS, err);
            return err;}

    return err;
}



//============================================================================
bool check_db()
{   
    uint32_t magic_read = 0;
    //Check if database is empty
    MXC_FLC_Read(TEST_ADDRESS, &magic_read, 4);
    PR_DEBUG("Magic Value at address 0x%x \tRead: 0x%x\n",
    TEST_ADDRESS, magic_read);
    
    return (magic_read == MAGIC);
}

//============================================================================
int add_person()
{
    char dummy_db[8][6] = {"AAAAAA", "BBBBBB", "CCCCCC", "DDDDDD", "EEEEEE", "FFFFFF", "GGGGGG", "HHHHHH"};
    int err = 0;


	while(!face_detected || !capture_key)
		{	
			face_detection();
            if (!record_mode) //If record mode is off, return to main menu
                return -1;
			//face_detected = 0;
		}

    capture_key = 0;

    face_id();

     PR_DEBUG("This is record\n");



     for (int i = 0; i < 16; i++) {

        PR_DEBUG("Writing buffer value 0x%x to address 0x%x...\n",  output_buffer[i], TEST_ADDRESS + ((i+1) * 4));

        err = MXC_FLC_Write32(TEST_ADDRESS + ((i+1) * 4), output_buffer[i]);

        if (err) {
        printf("Failed to write value to 0x%x with error code %i!\n", TEST_ADDRESS + ((i+1) * 4), err);
        return err; }
    }

    return err;

}

//============================================================================
void flash_to_cnn(int id)
{   
    uint32_t data, val;
    volatile uint32_t *kernel_addr, *ptr;
    uint32_t readval = 0;

    uint32_t kernel_buffer[4];
    uint32_t write_buffer[3];
    uint32_t emb_buffer[16];
    uint32_t emb;


    int block_id = id / 9;
    int block_offset = id % 9;
    int write_offset = 8 - block_offset; // reverse order for each block;

    PR_DEBUG("Block ID: %d\tBlock Offset: %d\tWrite Offset: %d\n", block_id, block_offset, write_offset);

    

    //Read the emb from flash
    for (int i = 0; i < 16; i++) {
        MXC_FLC_Read(TEST_ADDRESS + ((i+1) * 4), &readval, 4);
        emb_buffer[i] = readval;
        PR_DEBUG("Read value 0x%x from address 0x%x\n", readval, TEST_ADDRESS + ((i+1) * 4));
    }

    //Write the kernel to CNN

    for (int base_id = 0; base_id < EMBEDDING_SIZE; base_id++){
    
        //emb = (emb_buffer[base_id/4] >> (8 * (base_id % 4))) & 0x000000FF;
        emb = (emb_buffer[base_id/4] << (8 * (3 - (base_id % 4)))) & 0xFF000000; // 0xYYZZWWXX -> 0xXX000000, 0xYY000000, 0xZZ000000, 0xWW000000
        PR_DEBUG("Emb value: 0x%x\n", emb);
        kernel_addr = (volatile uint32_t *)(baseaddr[base_id] + block_id * 4);
        

        //Read the kernel from CNN first
        ptr = (volatile uint32_t *)(((uint32_t)kernel_addr & 0xffffe000) | (((uint32_t)kernel_addr & 0x1fff) << 2));
        

        for (int i = 0; i < 4; i++) {
            kernel_buffer[i] = *ptr;
            PR_DEBUG("Read value 0x%x from address 0x%x\n", kernel_buffer[i], (ptr));
            ptr++;

        }

        if (write_offset == 0){

            write_buffer[0] = emb | (kernel_buffer[1] >> 8); // kernel buffer 0 is always in a shape of 0x000000XX
        }
        else if (write_offset == 1){
            kernel_buffer[1] = ((kernel_buffer[1] & 0x00FFFFFF) | emb);
            PR_DEBUG("Kernel buffer 1: 0x%x\n", kernel_buffer[1]);
        }
        else if (write_offset == 2){
            kernel_buffer[1] = ((kernel_buffer[1] & 0xFF00FFFF) | (emb >> 8));
        }
        else if (write_offset == 3){
            kernel_buffer[1] = ((kernel_buffer[1] & 0xFFFF00FF) | (emb >> 16));
        }
        else if (write_offset == 4){
            kernel_buffer[1] = ((kernel_buffer[1] & 0xFFFFFF00) | (emb>> 24));
        }
        else if (write_offset == 5){
            kernel_buffer[2] = ((kernel_buffer[1] & 0x00FFFFFF) | emb);
        }
        else if (write_offset == 6){
            kernel_buffer[2] = ((kernel_buffer[1] & 0xFF00FFFF) | (emb >> 8));
        }
        else if (write_offset == 7){
            kernel_buffer[2] = ((kernel_buffer[1] & 0xFFFF00FF) | (emb >> 16));
        }
        else if (write_offset == 8){
            kernel_buffer[2] = ((kernel_buffer[1] & 0xFFFFFF00) | (emb >> 24));
        }


        /*
        else{
            kernel_buffer[((write_offset - 1) / 4) + 1] = ((kernel_buffer[((write_offset - 1) / 4) + 1] & (0xFFFFFF00 << ((3 - ((write_offset -1) % 4)) * 8))) | (emb << ((3 - ((write_offset -1) % 4)) * 8)));
        }*/
        
        *((volatile uint8_t *) ((uint32_t) kernel_addr | 1)) = 0x01; // Set address

        if (write_offset != 0){    
        write_buffer[0] = (kernel_buffer[0] << 24) | (kernel_buffer[1] >> 8); // kernel buffer 0 is always in a shape of 0x000000XX
        PR_DEBUG("Write buffer 0: 0x%x\n", write_buffer[0]); }

        write_buffer[1] = (kernel_buffer[1] << 24) | (kernel_buffer[2] >> 8);
        PR_DEBUG("Write buffer 1: 0x%x\n", write_buffer[1]);
        write_buffer[2] = (kernel_buffer[2] << 24) | (kernel_buffer[3] >> 8);
        PR_DEBUG("Write buffer 2: 0x%x\n", write_buffer[2]);

        // 4 is always empty, so we don't need to write it
        for (int i = 0; i < 3; i++) {

            PR_DEBUG("Writing value 0x%x to address 0x%x\n", write_buffer[i], kernel_addr);
            *kernel_addr++ = write_buffer[i];
            

        }

    }

}


//============================================================================
void reload_cnn()
{
    volatile uint32_t *kernel_addr;

    uint32_t readval = 0;
    int dummy = 0;
    int len = 0;

    //just load one for now
    
    
    PR_DEBUG("Kernel address: 0x%x\n", kernel_addr);
    

    for (uint32_t addr = TEST_ADDRESS+4; addr < TEST_ADDRESS + 17*4; addr += 4) {
        kernel_addr = (volatile uint32_t *) baseaddr[dummy++];
        *((volatile uint8_t *) ((uint32_t) kernel_addr | 1)) = 0x01; // Set address

        MXC_FLC_Read(addr, &readval, 4);
        //*kernel_addr = (readval) & 0xFF000000;
        //*kernel_addr++ = 0x00000000;
        len = 3;
        while (len-- > 0)   
        *kernel_addr++ = (readval << 24);

        //PR_DEBUG("kernel_addr 0x%x", kernel_addr);
        //PR_DEBUG("readval 0x%x", (readval) & 0xFF000000); 

        kernel_addr = (volatile uint32_t *) baseaddr[dummy++];
        *((volatile uint8_t *) ((uint32_t) kernel_addr | 1)) = 0x01; // Set address

        //*kernel_addr = (readval << 8) & 0xFF000000;
        len = 3;
        while (len-- > 0)   
        *kernel_addr++ = (readval << 16);
        //PR_DEBUG("kernel_addr 0x%x", kernel_addr);
        //PR_DEBUG("readval 0x%x", (readval << 8) & 0xFF000000); 

        kernel_addr = (volatile uint32_t *) baseaddr[dummy++];
        *((volatile uint8_t *) ((uint32_t) kernel_addr | 1)) = 0x01; // Set address

        //*kernel_addr = (readval << 16) & 0xFF000000;
        len = 3;
        while (len-- > 0)   
        *kernel_addr++ = (readval << 8);
        //PR_DEBUG("kernel_addr 0x%x", kernel_addr);
        //PR_DEBUG("readval 0x%x", (readval << 16) & 0xFF000000); 

        kernel_addr = (volatile uint32_t *) baseaddr[dummy++];
        *((volatile uint8_t *) ((uint32_t) kernel_addr | 1)) = 0x01; // Set address

        //*kernel_addr = (readval << 24) & 0xFF000000;
        len = 3;
        while (len-- > 0)   
        *kernel_addr++ = (readval);
        //PR_DEBUG("kernel_addr 0x%x", kernel_addr);
        //PR_DEBUG("readval 0x%x", (readval << 24) & 0xFF000000); 
        
        
        /*
        PR_DEBUG("Set Addr Pointer:%x \n", ((volatile uint8_t *) ((uint32_t) kernel_addr | 1)));
        PR_DEBUG("Kernel address: 0x%x, Kernel value:%x\n", kernel_addr, *kernel_addr);
        
        PR_DEBUG("After `` Kernel address: 0x%x, Kernel value:%x\n", kernel_addr, *kernel_addr);
        kernel_addr++;
        PR_DEBUG("Emb at address 0x%x \tRead: 0x%x\n",
                addr, readval); } */ }
        
        
    

}



void setup_irqs(void)
{
    /*
    All functions modifying flash contents are set to execute out of RAM
    with the (section(".flashprog")) attribute.  Therefore,
    
    If:
    - An FLC function is in the middle of execution (from RAM)
    ... and...
    - An interrupt triggers an ISR which executes from Flash

    ... Then a hard fault will be triggered.  
    
    FLC functions should be:
    1) Executed from a critical code block (interrupts disabled)
    or
    2) ISRs should be set to execute out of RAM with NVIC_SetRAM()

    This example demonstrates method #1.  Any code modifying
    flash is executed from a critical block, and the FLC
    interrupts will trigger afterwards.
    */

    // NVIC_SetRAM(); // Execute ISRs out of SRAM (for use with #2 above)
    MXC_NVIC_SetVector(FLC0_IRQn, FLC0_IRQHandler); // Assign ISR
    NVIC_EnableIRQ(FLC0_IRQn); // Enable interrupt

    __enable_irq();

    // Clear and enable flash programming interrupts
    MXC_FLC_EnableInt(MXC_F_FLC_INTR_DONEIE | MXC_F_FLC_INTR_AFIE);
    isr_flags = 0;
    isr_cnt = 0;
}

int record(void)
{   

    // First capture a name from touchscreen
    char name[20];
    PR_DEBUG("Enter name: ");


    setup_irqs(); // See notes in function definition

    /*
    Disable the instruction cache controller (ICC).

    Any code that modifies flash contents should disable the ICC,
    since modifying flash contents may invalidate cached instructions.
    */
    MXC_ICC_Disable(MXC_ICC0);

    //read database from flash and check magic value

    int err = 0;
    uint32_t magic_read = 0;
    /*
    err = MXC_FLC_PageErase(TEST_ADDRESS);
            if (err) {
            printf("Failed with error code %i\n", TEST_ADDRESS, err);
            return err; }*/

    
    if (check_db())
    {  

    PR_DEBUG("Magic Matched, Database found\n");
   
    }

    //if magic value does not match, initialize flash

    else 
        {
           err = init_db();
           if (err) {
            printf("Failed to initialize database", err);
            return err; }        
           
        }

    //if magic value matches, check if name already exists



    //if name exists, ask user to enter a different name
    
    //if name does not exist, proceed to capture face

    err = add_person();
    if (err == -1) { //TODO: change this to a way to exit to main menu
            printf("Exiting to main menu", err);
            return err; }
    else if (err != 0) {
            printf("Failed to add person", err);
            return err; }  

    //capture face

    //extract embedding

    //write embedding to flash

     PR_DEBUG("This is record\n");


    flash_to_cnn(17); // Load a single person's embeddings into CNN_3

    //reload_cnn();
    //cnn_verify_weights();
    //cnn_3_configure(); // Configure CNN_3 layers

    //write name to flash

    //write number of embeddings to flash

    //write number of subjects to flash

    //write length of embeddings to flash

    //write length of names to flash

    



    


    // Re-enable the ICC
    MXC_ICC_Enable(MXC_ICC0);
        printf("Successfully verified test pattern!\n\n");
        return err;
    
       
}
