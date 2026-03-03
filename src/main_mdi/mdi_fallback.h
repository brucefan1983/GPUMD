/*
   MDI - MolSSI Driver Interface
   https://molssi.org/, Molecular Sciences Software Institute

   This file is a simplified stub header providing MDI API declarations
   for compilation when the full MDI library is not available.
   
   This stub is derived from the original MDI library header:
   https://github.com/MolSSI-MDI/MDI_Library
   
   Original MDI Library Copyright:
   Copyright 2020 Taylor Barnes and contributors
   BSD-3-Clause license
   
   This stub version contributed by: Jaafar Mehrez (2026)
   Email: jaafarmehrez@sjtu.edu.cn, jaafar@hpqc.org
*/

#ifndef MDI_LIBRARY
#define MDI_LIBRARY

#ifdef __cplusplus
extern "C" {
#endif

/* ensure that symbols are exported to Windows .dll files */
#ifdef _WIN32
#define DllExport __declspec(dllexport)
#else
#define DllExport
#endif

/* type of an MDI communicator handle */
typedef int MDI_Comm;

/* type of an MDI datatype handle */
typedef int MDI_Datatype;

/* MDI version numbers */
DllExport extern const int MDI_MAJOR_VERSION;
DllExport extern const int MDI_MINOR_VERSION;
DllExport extern const int MDI_PATCH_VERSION;

/* length of an MDI command in characters */
DllExport extern const int MDI_COMMAND_LENGTH;

/* length of an MDI name in characters */
DllExport extern const int MDI_NAME_LENGTH;

/* length of an MDI label in characters */
DllExport extern const int MDI_LABEL_LENGTH;

/* value of a null communicator */
DllExport extern const MDI_Comm MDI_COMM_NULL;

/* MDI data types */
DllExport extern const int MDI_INT;
DllExport extern const int MDI_INT8_T;
DllExport extern const int MDI_INT16_T;
DllExport extern const int MDI_INT32_T;
DllExport extern const int MDI_INT64_T;
DllExport extern const int MDI_UINT8_T;
DllExport extern const int MDI_UINT16_T;
DllExport extern const int MDI_UINT32_T;
DllExport extern const int MDI_UINT64_T;
DllExport extern const int MDI_DOUBLE;
DllExport extern const int MDI_CHAR;
DllExport extern const int MDI_FLOAT;
DllExport extern const int MDI_BYTE;

/* MDI communication types */
DllExport extern const int MDI_TCP;
DllExport extern const int MDI_MPI;
DllExport extern const int MDI_LINK;
DllExport extern const int MDI_PLUGIN;
DllExport extern const int MDI_TEST;

/* MDI role types */
DllExport extern const int MDI_DRIVER;
DllExport extern const int MDI_ENGINE;

/* functions for handling MDI communication */
DllExport int MDI_Init(int* argc, char*** argv);
DllExport int MDI_Initialized(int* flag);
DllExport int MDI_Check_for_communicator(int* flag);
DllExport int MDI_Accept_Communicator(MDI_Comm* comm);
DllExport int MDI_Accept_communicator(MDI_Comm* comm);
DllExport int MDI_Send(const void* buf, int count, MDI_Datatype datatype, MDI_Comm comm);
DllExport int MDI_Recv(void* buf, int count, MDI_Datatype datatype, MDI_Comm comm);
DllExport int MDI_Send_Command(const char* buf, MDI_Comm comm);
DllExport int MDI_Send_command(const char* buf, MDI_Comm comm);
DllExport int MDI_Recv_Command(char* buf, MDI_Comm comm);
DllExport int MDI_Recv_command(char* buf, MDI_Comm comm);
DllExport int MDI_Conversion_Factor(const char* in_unit, const char* out_unit, double* conv);
DllExport int MDI_Conversion_factor(const char* in_unit, const char* out_unit, double* conv);
DllExport int MDI_Get_Role(int* role);
DllExport int MDI_Get_role(int* role);
DllExport int MDI_Get_method(int* role, MDI_Comm comm);
DllExport int MDI_Get_communicator(MDI_Comm* comm, int index);
DllExport int MDI_String_to_atomic_number(const char* element_symbol, int* atomic_number);

/* functions for managing Nodes, Commands, and Callbacks */
DllExport int MDI_Register_Node(const char* node_name);
DllExport int MDI_Register_node(const char* node_name);
DllExport int MDI_Check_Node_Exists(const char* node_name, MDI_Comm comm, int* flag);
DllExport int MDI_Check_node_exists(const char* node_name, MDI_Comm comm, int* flag);
DllExport int MDI_Get_NNodes(MDI_Comm comm, int* nnodes);
DllExport int MDI_Get_nnodes(MDI_Comm comm, int* nnodes);
DllExport int MDI_Get_Node(int index, MDI_Comm comm, char* name);
DllExport int MDI_Get_node(int index, MDI_Comm comm, char* name);
DllExport int MDI_Register_Command(const char* node_name, const char* command_name);
DllExport int MDI_Register_command(const char* node_name, const char* command_name);
DllExport int
MDI_Check_Command_Exists(const char* node_name, const char* command_name, MDI_Comm comm, int* flag);
DllExport int
MDI_Check_command_exists(const char* node_name, const char* command_name, MDI_Comm comm, int* flag);
DllExport int MDI_Get_NCommands(const char* node_name, MDI_Comm comm, int* ncommands);
DllExport int MDI_Get_ncommands(const char* node_name, MDI_Comm comm, int* ncommands);
DllExport int MDI_Get_Command(const char* node_name, int index, MDI_Comm comm, char* name);
DllExport int MDI_Get_command(const char* node_name, int index, MDI_Comm comm, char* name);
DllExport int MDI_Register_Callback(const char* node_name, const char* callback_name);
DllExport int MDI_Register_callback(const char* node_name, const char* callback_name);
DllExport int MDI_Check_Callback_Exists(
  const char* node_name, const char* callback_name, MDI_Comm comm, int* flag);
DllExport int MDI_Check_callback_exists(
  const char* node_name, const char* callback_name, MDI_Comm comm, int* flag);
DllExport int MDI_Get_NCallbacks(const char* node_name, MDI_Comm comm, int* ncallbacks);
DllExport int MDI_Get_ncallbacks(const char* node_name, MDI_Comm comm, int* ncallbacks);
DllExport int MDI_Get_Callback(const char* node_name, int index, MDI_Comm comm, char* name);
DllExport int MDI_Get_callback(const char* node_name, int index, MDI_Comm comm, char* name);

/* functions for handling MPI in combination with MDI */
DllExport int MDI_MPI_get_world_comm(void* world_comm);
DllExport int MDI_MPI_set_world_comm(void* world_comm);

#ifdef __cplusplus
}
#endif

#endif
