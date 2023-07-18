/*******************************************************************************
 * Copyright (C) 2022 Maxim Integrated Products, Inc., All rights Reserved.
 *
 * This software is protected by copyright laws of the United States and
 * of foreign countries. This material may also be protected by patent laws
 * and technology transfer regulations of the United States and of foreign
 * countries. This software is furnished under a license agreement and/or a
 * nondisclosure agreement and may only be used or reproduced in accordance
 * with the terms of those agreements. Dissemination of this information to
 * any party or parties not specified in the license agreement and/or
 * nondisclosure agreement is expressly prohibited.
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
 *******************************************************************************/
/********************************* DEFINES ******************************/

#include <stdio.h>
#include <stdint.h>


#define BIT_SET(a, b) ((a) |= (1U << (b)))
#define BIT_CLEAR(a, b) ((a) &= ~(1U << (b)))

#define SSD1306_SET_BUFFER_PIXEL_UTIL(buf, buf_w, buf_max, x, y, color, opa) \
    uint16_t byte_index = x + ((y >> 3) * buf_w);                            \
    uint8_t bit_index = y & 0x7;                                             \
    if (byte_index >= buf_max) {                                             \
        return;                                                              \
    }                                                                        \
                                                                             \
    if (color == 0 && opa) {                                                 \
        BIT_SET(buf[byte_index], bit_index);                                 \
    } else {                                                                 \
        BIT_CLEAR(buf[byte_index], bit_index);                               \
    }


void ssd1306_set_buffer_pixel_util(uint8_t *buf, uint16_t buf_w, uint32_t buf_max, uint16_t x,
                                   uint16_t y, uint8_t color, uint8_t is_opaque);
void draw_obj_rect(float* xy, uint32_t w, uint32_t h);
