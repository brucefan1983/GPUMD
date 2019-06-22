/*
    Copyright 2017 Zheyong Fan, Ville Vierimaa, Mikko Ervasti, and Ari Harju
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/


/*----------------------------------------------------------------------------80
Dump atom positions in netcdf compatible format.
------------------------------------------------------------------------------*/

//#ifdef USE_NETCDF

#include <unistd.h>
#include "dump_netcdf.cuh"

/* Handle errors by printing an error message and exiting with a
 * non-zero status. */
#define ERRCODE 2
#define ERR(e) {printf("Error: %s\n", nc_strerror(e)); exit(ERRCODE);}
#define NC_CHECK(s) {if(s != NC_NOERR) ERR(s);}

//TODO first implement basic open/write/close in the initialize function to
// see if I can get it to work at all

void DUMP_NETCDF::initialize(char *input_dir)
{
	strcpy(file_position, input_dir);
	strcat(file_position, "/movie.nc");
	if (access(file_position, F_OK != -1)) //check for file existence
	{
		// file exists, open file
		NC_CHECK(nc_open(file_position, NC_WRITE, &ncid));
	}
	else
	{
		// file does not exist, create file
		NC_CHECK(nc_create(file_position, NC_64BIT_DATA, &ncid));
	}

	NC_CHECK(nc_put_att_text(ncid, NC_GLOBAL, "program", 5, "GPUMD"));
	NC_CHECK(nc_put_att_text(ncid, NC_GLOBAL, "programVersion",
			strlen(GPUMD_VERSION), GPUMD_VERSION));

	NC_CHECK(nc_close(ncid));
}

void DUMP_NETCDF::finalize()
{
	//TODO do something
}

void DUMP_NETCDF::dump(Atom *atom, int step)
{
	//TODO do something
}

//#endif
