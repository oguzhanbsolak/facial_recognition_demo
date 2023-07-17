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
#define DB_ADDRESS (MXC_FLASH_MEM_BASE + MXC_FLASH_MEM_SIZE) - (1 * MXC_FLASH_PAGE_SIZE)

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
        NNNNNNNN
        NNNNIIIL        
        EEEEEEEE
        ........
        ........
        ........
        NNNNNNNN
        NNNNIIIL        
        EEEEEEEE
        ........
        N : Name of the person (6 bytes)
        I : Unique ID of the person (12 bits)
        L : Embeddings count of the person (4 bits)
        E : Embeddings of the person (L * 64 bytes)

Example Status Field :
        TTTTTTTT
        CCCCCCCC 
        T : Total number of people in the database (32 bits)
        C : Total embeddings count in the database (32 bits)
*/

struct person
{
    char name[6];
    uint32_t id; 
    uint32_t embeddings_count;
    uint32_t db_embeddings_count;
};

typedef struct person Person;


/***** Globals *****/
volatile uint32_t isr_cnt;
volatile uint32_t isr_flags;
extern volatile int32_t output_buffer[16];
extern unsigned int touch_x, touch_y;
extern volatile uint8_t face_detected;
extern volatile uint8_t capture_key;
extern volatile uint8_t record_mode;
static const uint32_t baseaddr[] = BASEADDR;

/***** Prototypes *****/
void get_status(Person *p);
int update_status(Person *p);
int update_info_field(Person *p);
void FLC0_IRQHandler();
int init_db();
int init_status();
int add_person(Person *p);
void flash_to_cnn(Person *p, uint32_t cnn_location);
void setup_irqs();
void read_db(Person *p);
bool check_db();



void read_db(Person *p)
{   
    // Description : Reads the database from flash and populates the Person structure, works in loop

    uint32_t info_address;
    uint32_t first_info;
    uint32_t second_info;

    info_address = (DB_ADDRESS + 4) + ((p->id - 1) * 4 * 2) + (p->db_embeddings_count * 64);

    MXC_FLC_Read(info_address, &first_info, 4);
    MXC_FLC_Read(info_address + 4, &second_info, 4);

    p->name[0] = (first_info >> 24) & 0xFF;
    p->name[1] = (first_info >> 16) & 0xFF;
    p->name[2] = (first_info >> 8) & 0xFF;
    p->name[3] = (first_info) & 0xFF;
    p->name[4] = (second_info >> 24) & 0xFF;
    p->name[5] = (second_info >> 16) & 0xFF;
    p->id = (second_info >> 4) & 0xFFF;
    p->embeddings_count = (second_info) & 0xF;
        

}


void init_cnn_from_flash()
{
    uint32_t location = 0;
    uint32_t counter = 0;

    if(!check_db())
    {
        PR_DEBUG("No database found, skipping CNN database initialization\n");
        return;
    }
    PR_DEBUG("Initializing CNN database from flash\n");
    Person total;
    Person *total_ptr = &total;
    get_status(total_ptr);

    Person p;
    Person *pptr = &p;

    pptr->id = 1;
    pptr->embeddings_count = 0;
    pptr->db_embeddings_count = 0;


    
    for (uint32_t i = 1; i < total_ptr->id; i++)
    {
        read_db(pptr); //Get the name, id and embeddings count of the person
        PR_DEBUG("Reading person %s with id %d and embeddings count %d\n", pptr->name, pptr->id, pptr->embeddings_count);
        counter = pptr->embeddings_count;
        pptr->embeddings_count = 0;
        for (uint32_t j = 0; j < counter; j++)
        {
            flash_to_cnn(pptr, location);
            location += 1;
            pptr->embeddings_count += 1;
        }
        pptr->db_embeddings_count = pptr->db_embeddings_count + counter;
        pptr->id += 1;
    }

}




void get_status(Person *p)
{
    uint32_t id;
    uint32_t count;

    MXC_FLC_Read(STATUS_ADDRESS, &id, 4);
    if (id == 0xFFFFFFFF) { // Flash is empty
        p->id = 1;
        p->embeddings_count = 0;
        p->db_embeddings_count = 0;
        return;
    }
    

    p->id = id + 1;

    MXC_FLC_Read(STATUS_ADDRESS + 4, &count, 4);
    p->embeddings_count = 0; // Initialize to 0 for every new person
    p->db_embeddings_count = count;
}

int update_status(Person *p)
{   
    int err = 0;
    err = init_status();
        if (err) {
        printf("Failed to initialize status", err);
        return err; }
    err = MXC_FLC_Write32(STATUS_ADDRESS, p->id);
        if (err) {
        printf("Failed to write status (ID)", err);
        return err; } 
    err = MXC_FLC_Write32(STATUS_ADDRESS + 4, p->db_embeddings_count);
        if (err) {
        printf("Failed to write status (Embeddings count)", err);
        return err; }
    return err;
}

int update_info_field(Person *p)
{   
    int err = 0;

    uint32_t first_info = 0x00000000;
    uint32_t second_info = 0x00000000;   
    uint32_t info_address;

    /*
    NNNNNNNN First info field
    NNNNIIIL Second info field
    */
 
    info_address = (DB_ADDRESS + 4) + ((p->id - 1) * 4 * 2) + (p->db_embeddings_count * 64); //TODO : Control total emb and emb in a more elegant way

    first_info = (p->name[0] << 24) | (p->name[1] << 16) | (p->name[2] << 8) | (p->name[3]);



    second_info = (p->name[4] << 24) | (p->name[5] << 16) | (p->id << 4) | (p->embeddings_count);

    err = MXC_FLC_Write32(info_address, first_info);
        if (err) {
        printf("Failed to write status to 0x%x, %d, (First info)", info_address, err);
        return err; }

    err = MXC_FLC_Write32(info_address + 4, second_info);
        if (err) {
        printf("Failed to write status to 0x%x, %d, (Second info)", info_address + 4, err);
        return err; }

    return err;
}


void FLC0_IRQHandler()
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
 printf("Erasing page 64 of flash (addr 0x%x)...\n", DB_ADDRESS);
            err = MXC_FLC_PageErase(DB_ADDRESS);
            if (err) {
            printf("Failed with error code %i\n", DB_ADDRESS, err);
            return err; }

            PR_DEBUG("Magic Value not matched, Initializing flash\n");
            err = MXC_FLC_Write32(DB_ADDRESS, MAGIC);

            if (err) {
            printf("Failed to write magic value to 0x%x with error code %i!\n", DB_ADDRESS, err);
            return err;}

    return err;
}

int init_status()
{
    int err = 0;
    printf("Erasing page 63 of flash (addr 0x%x)...\n", STATUS_ADDRESS);
            err = MXC_FLC_PageErase(STATUS_ADDRESS);
            if (err) {
            printf("Failed with error code %i\n", STATUS_ADDRESS, err);
            return err; }


    return err;
}


//============================================================================
bool check_db()
{   
    uint32_t magic_read = 0;
    //Check if database is empty
    MXC_FLC_Read(DB_ADDRESS, &magic_read, 4);
    PR_DEBUG("Magic Value at address 0x%x \tRead: 0x%x\n",
    DB_ADDRESS, magic_read);
    
    return (magic_read == MAGIC);
}

//============================================================================
int add_person(Person *p)
{
    int err = 0;

    face_detected = 0;

    if (p->embeddings_count == 0) {
        PR_DEBUG("Enter name: ");

        scanf("%5s", p->name);
        PR_DEBUG("Name entered: %s\n", p->name); //TODO:Get the name from TS
    }   



	while(!face_detected || !capture_key)
		{	
			face_detection();

			//face_detected = 0;
		}

    capture_key = 0;

    face_id();

     PR_DEBUG("This is record\n");

     //Calculate the write address 4 bytes for magic key, 8 bytes for each person, 64 bytes for each embedding
    PR_DEBUG("p.id %d" , p->id);
    PR_DEBUG("p.embeddings_count %d", p->embeddings_count);
    PR_DEBUG("Total embeddings_count %d", p->db_embeddings_count); 
    uint32_t write_address = (DB_ADDRESS + 4) + ((p->id - 1) * 4 * 2) + ((p->embeddings_count + p->db_embeddings_count) * 64);



     for (int i = 0; i < 16; i++) {

        PR_DEBUG("Writing buffer value 0x%x to address 0x%x...\n",  output_buffer[i], write_address + ((i+2) * 4)); // 2 for the name and length field
        

        err = MXC_FLC_Write32(write_address + ((i+2) * 4), output_buffer[i]);

        if (err) {
        printf("Failed to write value to 0x%x with error code %i!\n", write_address + ((i+2) * 4), err);
        return err; }
    }
    


    flash_to_cnn(p, (p->embeddings_count + p->db_embeddings_count)); // Load a single embedding into CNN_3
    p->embeddings_count += 1; // TODO: Check this later for add person, add embedding logic
    //p->db_embeddings_count += 1;

    record_mode = 0;
    PR_DEBUG("To continue to capture press P2.6, to return to main menu press P2.7\n");
    while(!capture_key)
		{	
            if (record_mode){ //If record mode is off, return to main menu

                err = update_info_field(p); //Update the information field
                
                p->db_embeddings_count = p->db_embeddings_count + p->embeddings_count;

                if (err) {
                printf("Failed to update info field with error code %i!\n", err);
                return err; }   

                err = -1; // -1 is the exit code
                return err;}
		}

    capture_key = 0;

    err = add_person(p);

    return err; //Never expect to reach here

}

//============================================================================
void flash_to_cnn(Person *p, uint32_t cnn_location)
{   
    volatile uint32_t *kernel_addr, *ptr;
    uint32_t readval = 0;

    uint32_t kernel_buffer[4];
    uint32_t write_buffer[3];
    uint32_t emb_buffer[16];
    uint32_t emb;
    uint32_t emb_addr;

    //Reload the latest emb
    //TODO: Rethink this logic
    int block_id = (cnn_location) / 9;
    int block_offset = (cnn_location) % 9;
    int write_offset = 8 - block_offset; // reverse order for each block;

    PR_DEBUG("Block ID: %d\tBlock Offset: %d\tWrite Offset: %d\n", block_id, block_offset, write_offset);

    emb_addr = (DB_ADDRESS + 4) + ((p->id) * 4 * 2) + ((p->embeddings_count + p->db_embeddings_count) * 64); // 4 bytes for magic key, 8 bytes for each person, 64 bytes for each embedding

    

    //Read the emb from flash
    for (int i = 0; i < 16; i++) {
        MXC_FLC_Read(emb_addr + i*4, &readval, 4);
        emb_buffer[i] = readval;
        PR_DEBUG("Read value 0x%x from address 0x%x\n", readval, emb_addr + i* 4);
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
            //PR_DEBUG("Kernel buffer 1: 0x%x\n", kernel_buffer[1]);
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
            kernel_buffer[2] = ((kernel_buffer[2] & 0x00FFFFFF) | emb);
        }
        else if (write_offset == 6){
            kernel_buffer[2] = ((kernel_buffer[2] & 0xFF00FFFF) | (emb >> 8));
        }
        else if (write_offset == 7){
            kernel_buffer[2] = ((kernel_buffer[2] & 0xFFFF00FF) | (emb >> 16));
        }
        else if (write_offset == 8){
            kernel_buffer[2] = ((kernel_buffer[2] & 0xFFFFFF00) | (emb >> 24));
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


void setup_irqs()
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

int record()
{   

    // First capture a name from touchscreen
    

    Person p;
    Person *pptr = &p;

    setup_irqs(); // See notes in function definition

    /*
    Disable the instruction cache controller (ICC).

    Any code that modifies flash contents should disable the ICC,
    since modifying flash contents may invalidate cached instructions.
    */
    MXC_ICC_Disable(MXC_ICC0);

    //read database from flash and check magic value

    int err = 0;
    /*
    err = MXC_FLC_PageErase(DB_ADDRESS);
            if (err) {
            printf("Failed with error code %i\n", DB_ADDRESS, err);
            return err; }*/

    
    if (check_db())
    {  

    PR_DEBUG("Magic Matched, Database found\n");
    //if magic value matches, get the latest ID from flash

    
    }

    //if magic value does not match, initialize flash

    else 
        {
           err = init_db();
           if (err) {
            printf("Failed to initialize database", err);
            return err; }
            err = init_status();
            if (err) {
            printf("Failed to initialize status", err);
            return err; }          
           
        }

    //if magic value matches, check if name already exists



    //if name exists, ask user to enter a different name
    
    //if name does not exist, proceed to capture face
    get_status(pptr);
    PR_DEBUG("Latest ID: %d\n", pptr->id);
    PR_DEBUG("Total embeddings: %d\n", pptr->db_embeddings_count);

    err = add_person(pptr);
    if (err == -1) { //TODO: change this to a way to exit to main menu
            err = update_status(pptr);
            if (err) {
            printf("Failed to update status", err);
            return err; }
            printf("Exiting to main menu", err);
            // Re-enable the ICC
            MXC_ICC_Enable(MXC_ICC0);
            printf("Successfully verified test pattern!\n\n");
            return err; }
    else if (err != 0) {
            printf("Failed to add person", err);
            return err; }  

    //capture face

    //extract embedding

    //write embedding to flash

     PR_DEBUG("This is record\n");


    


    //Update Status in flash
    

    //cnn_verify_weights();
    //cnn_3_configure(); // Configure CNN_3 layers

    //write name to flash

    //write number of embeddings to flash

    //write number of subjects to flash

    //write length of embeddings to flash

    //write length of names to flash

    



    


    
        return err;
    
       
}
