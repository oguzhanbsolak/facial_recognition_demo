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

/**
 * @file    main.c
 * @brief   Facial Recognition MAX78002 Evkit Demo
 *
 * @details
 *
 */

#define S_MODULE_NAME "main"

/***** Includes *****/
#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include "board.h"
#include "mxc.h"
#include "mxc_device.h"
#include "mxc_delay.h"
#include "camera.h"
#include "icc.h"
#include "rtc.h"
#include "lp.h"
#include "cnn_1.h"
#include "cnn_2.h"
#include "cnn_3.h"
#include "MAXCAM_Debug.h"
#include "facedetection.h"
#include "post_process.h"
#include "faceID.h"
#include "record.h"
#include "gpio.h"
#include "pb.h"

#define CONSOLE_BAUD  115200

#define MXC_GPIO_PORT_INTERRUPT_IN MXC_GPIO2
#define MXC_GPIO_PIN_INTERRUPT_IN MXC_GPIO_PIN_7

#define MXC_GPIO_PORT_INTERRUPT_IN_2 MXC_GPIO2
#define MXC_GPIO_PIN_INTERRUPT_IN_2 MXC_GPIO_PIN_6

extern volatile uint8_t face_detected;
volatile uint8_t record_mode = 0;
volatile uint8_t capture_key = 0;

#if 0 // Custom camera settings
static const uint8_t camera_settings[][2] = {
	{0x0e, 0x08}, // Sleep mode
	{0x69, 0x52}, // BLC window selection, BLC enable (default is 0x12)
	{0x1e, 0xb3}, // AddLT1F (default 0xb1)
	{0x48, 0x42},
	{0xff, 0x01}, // Select MIPI register bank
	{0xb5, 0x30},
	{0xff, 0x00}, // Select system control register bank
	{0x16, 0x03}, // (default)
	{0x62, 0x10}, // (default)
	{0x12, 0x01}, // Select Bayer RAW
	{0x17, 0x65}, // Horizontal Window Start Point Control (LSBs), default is 0x69
	{0x18, 0xa4}, // Horizontal sensor size (default)
	{0x19, 0x0c}, // Vertical Window Start Line Control (default)
	{0x1a, 0xf6}, // Vertical sensor size (default)
	{0x37, 0x04}, // PCLK is double system clock (default is 0x0c)
	{0x3e, 0x20}, // (default)
	{0x81, 0x3f}, // sde_en, uv_adj_en, scale_v_en, scale_h_en, uv_avg_en, cmx_en
	{0xcc, 0x02}, // High 2 bits of horizontal output size (default)
	{0xcd, 0x80}, // Low 8 bits of horizontal output size (default)
	{0xce, 0x01}, // Ninth bit of vertical output size (default)
	{0xcf, 0xe0}, // Low 8 bits of vertical output size (default)
	{0x82, 0x01}, // 01: Raw from CIP (default is 0x00)
	{0xc8, 0x02},
	{0xc9, 0x80},
	{0xca, 0x01},
	{0xcb, 0xe0},
	{0xd0, 0x28},
	{0x0e, 0x00}, // Normal mode (not sleep mode)
	{0x70, 0x00},
	{0x71, 0x34},
	{0x74, 0x28},
	{0x75, 0x98},
	{0x76, 0x00},
	{0x77, 0x64},
	{0x78, 0x01},
	{0x79, 0xc2},
	{0x7a, 0x4e},
	{0x7b, 0x1f},
	{0x7c, 0x00},
	{0x11, 0x01}, // CLKRC, Internal clock pre-scalar divide by 2 (default divide by 1)
	{0x20, 0x00}, // Banding filter (default)
	{0x21, 0x57}, // Banding filter (default is 0x44)
	{0x50, 0x4d},
	{0x51, 0x40}, // 60Hz Banding AEC 8 bits (default 0x80)
	{0x4c, 0x7d},
	{0x0e, 0x00},
	{0x80, 0x7f},
	{0x85, 0x00},
	{0x86, 0x00},
	{0x87, 0x00},
	{0x88, 0x00},
	{0x89, 0x2a},
	{0x8a, 0x22},
	{0x8b, 0x20},
	{0xbb, 0xab},
	{0xbc, 0x84},
	{0xbd, 0x27},
	{0xbe, 0x0e},
	{0xbf, 0xb8},
	{0xc0, 0xc5},
	{0xc1, 0x1e},
	{0xb7, 0x05},
	{0xb8, 0x09},
	{0xb9, 0x00},
	{0xba, 0x18},
	{0x5a, 0x1f},
	{0x5b, 0x9f},
	{0x5c, 0x69},
	{0x5d, 0x42},
	{0x24, 0x78}, // AGC/AEC
	{0x25, 0x68}, // AGC/AEC
	{0x26, 0xb3}, // AGC/AEC
	{0xa3, 0x0b},
	{0xa4, 0x15},
	{0xa5, 0x29},
	{0xa6, 0x4a},
	{0xa7, 0x58},
	{0xa8, 0x65},
	{0xa9, 0x70},
	{0xaa, 0x7b},
	{0xab, 0x85},
	{0xac, 0x8e},
	{0xad, 0xa0},
	{0xae, 0xb0},
	{0xaf, 0xcb},
	{0xb0, 0xe1},
	{0xb1, 0xf1},
	{0xb2, 0x14},
	{0x8e, 0x92},
	{0x96, 0xff},
	{0x97, 0x00},
	{0x14, 0x3b},	// AGC value, manual, set banding (default is 0x30)
	{0x0e, 0x00},
	{0x0c, 0xd6},
	{0x82, 0x3},
	{0x11, 0x00},	// Set clock prescaler
    {0x12, 0x6},
    {0x61, 0x0},
    {0x64, 0x11},
    {0xc3, 0x80},
    {0x81, 0x3f},
    {0x16, 0x3},
    {0x37, 0xc},
    {0x3e, 0x20},
    {0x5e, 0x0},
    {0xc4, 0x1},
    {0xc5, 0x80},
    {0xc6, 0x1},
    {0xc7, 0x80},
    {0xc8, 0x2},
    {0xc9, 0x80},
    {0xca, 0x1},
    {0xcb, 0xe0},
    {0xcc, 0x0},
    {0xcd, 0x40},	// Default to 64 line width
    {0xce, 0x0},
    {0xcf, 0x40},	// Default to 64 lines high
    {0x1c, 0x7f},
    {0x1d, 0xa2},
	{0xee, 0xee}  // End of register list marker 0xee
};
#endif

mxc_uart_regs_t* CommUart;
unsigned int touch_x, touch_y;
int font_1 = (int)&SansSerif16x16[0];


void gpio_isr(void *cbdata)
{
	record_mode = 1; //Toggle record mode
	PR_DEBUG("TOGGLED record_mode = %d\n", record_mode);
}

void gpio_isr_2(void *cbdata)
{
	capture_key = 1;
}

#ifdef TFT_ENABLE
area_t area = {50, 290, 180, 30};
#endif
// *****************************************************************************
void TFT_Print(char *str, int x, int y, int font, int length)
{
    text_t text = { .data = str, .len = length };

	y = TFT_HEIGHT - y; //Rotate 90 degrees
    MXC_TFT_PrintFont(y, x, font, &text, NULL); //Rotate 90 degrees

}


void print_xy(unsigned int x, unsigned int y)
{
    char buf[16];

    MXC_TFT_ClearScreen();
    TFT_Print(buf, x, y, font_1, snprintf(buf, sizeof(buf), "(%u,%u)", x, y));
}


void WUT_IRQHandler()
{
    MXC_WUT_IntClear();
}

uint32_t ticks_1;
uint32_t ticks_2;
mxc_wut_cfg_t cfg;

int main(void)
{
    int ret = 0;
    int slaveAddress;
    int id;
    int dma_channel;
	int key;
	mxc_uart_regs_t* ConsoleUart;

    /* Enable cache */
    MXC_ICC_Enable(MXC_ICC0);

    // Switch to 120 MHz clock
    MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IPO);
    MXC_GCR->ipll_ctrl |= MXC_F_GCR_IPLL_CTRL_EN; // Enable IPLL
    SystemCoreClockUpdate();

    ConsoleUart = MXC_UART_GET_UART(CONSOLE_UART);

    if ((ret = MXC_UART_Init(ConsoleUart, CONSOLE_BAUD, MXC_UART_IBRO_CLK)) != E_NO_ERROR) {
        PR_ERR("UART1 Init Error: %d\n", ret);
		return ret;
    }

    printf("Waiting...\n");

    // DO NOT DELETE THIS LINE:
    MXC_Delay(SEC(2)); // Let debugger interrupt if needed

    // Enable peripheral, enable CNN interrupt, turn on CNN clock
    // CNN clock: PLL (200 MHz) div 4
    //cnn_2_enable(MXC_S_GCR_PCLKDIV_CNNCLKSEL_IPLL, MXC_S_GCR_PCLKDIV_CNNCLKDIV_DIV4);
  	cnn_1_enable(MXC_S_GCR_PCLKDIV_CNNCLKSEL_IPLL, MXC_S_GCR_PCLKDIV_CNNCLKDIV_DIV4);
    cnn_1_init(); // Bring state machine into consistent state
    cnn_1_load_weights(); // Load kernels of CNN_1
    cnn_1_load_bias(); // Load bias data of CNN_1
    cnn_1_configure(); // Configure CNN_1 layers

    cnn_2_load_weights(); // Load kernels of CNN_2
    cnn_2_load_bias(); // Load bias data of CNN_2
    cnn_2_configure(); // Configure CNN_2 layers

	cnn_3_load_weights(); // Load kernels of CNN_3
	//TODO: immplement reload_cnn() in a way that it includes new people in addition to the default ones
	//reload_cnn(); // Reload CNN_3 weights with new data from flash
  	cnn_3_load_bias(); // Load bias data of CNN_3
  	cnn_3_configure(); // Configure CNN_3 layers

	// Initialize Database from flash
	init_cnn_from_flash();


    /* Initialize RTC */
    MXC_RTC_Init(0, 0);
    MXC_RTC_Start();

    // Initialize DMA for camera interface
    MXC_DMA_Init();
    dma_channel = MXC_DMA_AcquireChannel();

    PR_DEBUG("\n\nMAX78002 Facial Recognition Demo\n");

    // Initialize the camera driver.
    camera_init(CAMERA_FREQ);

    // Obtain the I2C slave address of the camera.
    slaveAddress = camera_get_slave_address();
    PR_DEBUG("Camera I2C slave address is %02x\n", slaveAddress);

    // Obtain the product ID of the camera.
    ret = camera_get_product_id(&id);

    if (ret != STATUS_OK) {
        PR_ERR("Error returned from reading camera id. Error %d\n", ret);
        return -1;
    }

    PR_DEBUG("Camera Product ID is %04x\n", id);

    // Obtain the manufacture ID of the camera.
    ret = camera_get_manufacture_id(&id);

    if (ret != STATUS_OK) {
        PR_ERR("Error returned from reading camera id. Error %d\n", ret);
        return -1;
    }

    PR_DEBUG("Camera Manufacture ID is %04x\n", id);

#if 0
    // set camera registers with custom values
	for (int i = 0; (camera_settings[i][0] != 0xee); i++) {
		camera_write_reg(camera_settings[i][0], camera_settings[i][1]);
	}
#endif
	
    // Setup the camera image dimensions, pixel format and data acquiring details.
    ret = camera_setup(IMAGE_XRES, IMAGE_YRES, PIXFORMAT_RGB565, FIFO_FOUR_BYTE, USE_DMA,
                       dma_channel);

    if (ret != STATUS_OK) {
        PR_ERR("Error returned from setting up camera. Error %d\n", ret);
        return -1;
    }
	PR_DEBUG("b flip");
	camera_set_vflip(0);
	PR_DEBUG("a flip");

#ifdef TFT_ENABLE
    /* Initialize TFT display */
    //MXC_TFT_Init(MXC_SPI0, 1, NULL, NULL);
	
	PR_DEBUG("TFT flip");
	MXC_TFT_Init(NULL, NULL);
	PR_DEBUG("TFT INIT");
    MXC_TFT_SetRotation(ROTATE_180);
    MXC_TFT_SetBackGroundColor(4);
    MXC_TFT_SetForeGroundColor(WHITE); // set font color to white
	#ifdef TS_ENABLE
		MXC_TS_Init();	
    	MXC_TS_Start();
		MXC_TS_AddButton(0, 0, 80, 80, 1);
		MXC_TS_AddButton(160, 0, 240, 80, 2);
	#else
		mxc_gpio_cfg_t gpio_interrupt;
		gpio_interrupt.port = MXC_GPIO_PORT_INTERRUPT_IN;
    	gpio_interrupt.mask = MXC_GPIO_PIN_INTERRUPT_IN;
    	gpio_interrupt.pad = MXC_GPIO_PAD_PULL_UP;
    	gpio_interrupt.func = MXC_GPIO_FUNC_IN;
    	gpio_interrupt.vssel = MXC_GPIO_VSSEL_VDDIOH;
    	MXC_GPIO_Config(&gpio_interrupt);
    	MXC_GPIO_RegisterCallback(&gpio_interrupt, gpio_isr, NULL);
    	MXC_GPIO_IntConfig(&gpio_interrupt, MXC_GPIO_INT_FALLING);
    	MXC_GPIO_EnableInt(gpio_interrupt.port, gpio_interrupt.mask);
    	NVIC_EnableIRQ(MXC_GPIO_GET_IRQ(MXC_GPIO_GET_IDX(MXC_GPIO_PORT_INTERRUPT_IN)));

		mxc_gpio_cfg_t gpio_interrupt_2;
		gpio_interrupt_2.port = MXC_GPIO_PORT_INTERRUPT_IN_2;
    	gpio_interrupt_2.mask = MXC_GPIO_PIN_INTERRUPT_IN_2;
    	gpio_interrupt_2.pad = MXC_GPIO_PAD_PULL_UP;
    	gpio_interrupt_2.func = MXC_GPIO_FUNC_IN;
    	gpio_interrupt_2.vssel = MXC_GPIO_VSSEL_VDDIOH;
    	MXC_GPIO_Config(&gpio_interrupt_2);
    	MXC_GPIO_RegisterCallback(&gpio_interrupt_2, gpio_isr_2, NULL);
    	MXC_GPIO_IntConfig(&gpio_interrupt_2, MXC_GPIO_INT_FALLING);
    	MXC_GPIO_EnableInt(gpio_interrupt_2.port, gpio_interrupt_2.mask);
    	NVIC_EnableIRQ(MXC_GPIO_GET_IRQ(MXC_GPIO_GET_IDX(MXC_GPIO_PORT_INTERRUPT_IN_2)));
	#endif

#endif

#ifdef LP_MODE_ENABLE
    /* Get ticks based on milliseconds */
    MXC_WUT_GetTicks(500, MXC_WUT_UNIT_MILLISEC, &ticks_1);
    MXC_WUT_GetTicks(100, MXC_WUT_UNIT_MILLISEC, &ticks_2);
    /* Configure structure for one shot timer to trigger in a number of ticks */
    cfg.mode    = MXC_WUT_MODE_ONESHOT;
    cfg.cmp_cnt = ticks_1;
    /* Init WakeUp Timer */
    MXC_WUT_Init(MXC_WUT_PRES_1);
    /* Config WakeUp Timer */
    MXC_WUT_Config(&cfg);
    /* Enable Alarm wakeup by WUT */
    MXC_LP_EnableWUTAlarmWakeup();
    /* Enable WakeUp Timer interrupt */
    NVIC_EnableIRQ(WUT_IRQn);
#endif

    while (1) {
		
		#ifdef TS_ENABLE //TODO: update here for record mode
		key = MXC_TS_GetKey();
		if (key == 1) {
			record_mode = 1;
		}

			/*if (MXC_TS_GetTSEvent()) {
            	MXC_TS_ClearTSEvent();
            	MXC_TS_GetXY(&touch_x, &touch_y); //TS Rotate 90 degree
				PR_DEBUG("Touch X: %d, Y: %d\n", touch_x, touch_y);

				if (touch_x > 0 && touch_x < 80 && touch_y > 0 && touch_y < 80) // button for now
				{ 
					record_mode = 1;
				}


            //print_xy(touch_x, touch_y);
        	}*/
		#endif 
		if (record_mode)
		{
			//printf("record mode\n");
			
			record();
			// Delay for 0.5 seconds before continuing
			//TODO: More elegant way to do this
			MXC_Delay(MXC_DELAY_MSEC(500));
			record_mode = 0;
		}
		else {

			face_detection();		

			if(face_detected)
			{	
				face_id();
			
				face_detected = 0;
			}

			//#if 1
			#ifdef TFT_ENABLE
			else
			{
            	MXC_TFT_ClearArea(&area, 4);
        	}
			#endif

		}
		

    }

    return 0;
}
