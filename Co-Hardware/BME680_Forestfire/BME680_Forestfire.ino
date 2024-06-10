/**
 * Copyright (C) 2021 Bosch Sensortec GmbH
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * 
 */

#include "Arduino.h"
#include "bme68xLibrary.h"

#define NEW_GAS_MEAS (BME68X_GASM_VALID_MSK | BME68X_HEAT_STAB_MSK | BME68X_NEW_DATA_MSK)

#ifndef PIN_CS
#define PIN_CS SS
#endif

String rx_buff;
bool rx_stat;
bool get_data;

void serialEvent() {
  rx_stat = true;
}

Bme68x bme;

/**
 * @brief Initializes the sensor and hardware settings
 */


void setup(void)
{
  rx_stat = false;
  get_data = false;

	SPI.begin();
	Serial.begin(115200); 

  delay(5000);
  Serial.println("Setup sensor");

	/* Initializes the sensor based on SPI library */
	bme.begin(PIN_CS, SPI);

	if(bme.checkStatus())
	{
		if (bme.checkStatus() == BME68X_ERROR)
		{
			Serial.println("Sensor error:" + bme.statusString());
			return;
		}
		else if (bme.checkStatus() == BME68X_WARNING)
		{
			Serial.println("Sensor Warning:" + bme.statusString());
		}
	}

	/* Set the default configuration for temperature, pressure and humidity */
	bme.setTPH();

	/* Heater temperature in degree Celsius */
	uint16_t tempProf[10] = { 100, 200, 320 };
	/* Heating duration in milliseconds */
	uint16_t durProf[10] = { 150, 150, 150 };

	bme.setSeqSleep(BME68X_ODR_250_MS);
	bme.setHeaterProf(tempProf, durProf, 3);
	bme.setOpMode(BME68X_SEQUENTIAL_MODE);

	Serial.println("TimeStamp(ms), Temperature(deg C), Pressure(Pa), Humidity(%), Gas resistance(ohm), Status, Gas index");
}

void loop(void)
{
	bme68xData data;
	uint8_t nFieldsLeft = 0;
  
  Serial.println("Test sending uart");
	delay(2000);

  if (rx_stat){
    rx_stat = false;
    // Get data from buffer
    if (Serial.available() > 0) {
    // read the incoming byte:
    rx_buff = Serial.readStringUntil('\n');
    Serial.print("I received: ");
    Serial.println(rx_buff);
    }

    if (rx_buff == "START"){
      get_data = true;
      Serial.println("start measuring data");
    }
  }
	if ((bme.fetchData() && get_data))
	{
		do
		{
      get_data = false;
			nFieldsLeft = bme.getData(data);
			//if (data.status == NEW_GAS_MEAS)
			{
				Serial.print(String(millis()) + ", ");
        Serial.print("temperature: "+String(data.temperature) + ", ");
				Serial.print("pressure: "+String(data.pressure) + ", ");
				Serial.print("humidity: "+String(data.humidity) + ", ");
				Serial.print("resistance: "+String(data.gas_resistance) + ", ");
				Serial.print("status: "+String(data.status, HEX) + ", ");
				Serial.println("gas index: "+String(data.gas_index));
				if(data.gas_index == 2) /* Sequential mode sleeps after this measurement */
					delay(250);
			}
		} while (nFieldsLeft);
	}
}
