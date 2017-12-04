{\rtf1\ansi\ansicpg1252\cocoartf1504\cocoasubrtf830
{\fonttbl\f0\fnil\fcharset0 Menlo-Regular;}
{\colortbl;\red255\green255\blue255;\red255\green255\blue255;\red178\green24\blue137;\red65\green182\blue69;
\red120\green109\blue196;\red219\green44\blue56;}
{\*\expandedcolortbl;;\csgenericrgb\c100000\c100000\c100000;\csgenericrgb\c69800\c9500\c53600;\csgenericrgb\c25500\c71400\c27000;
\csgenericrgb\c46900\c42600\c77000;\csgenericrgb\c85900\c17100\c21900;}
\margl1440\margr1440\vieww25400\viewh16000\viewkind0
\deftab890
\pard\tx890\pardeftab890\pardirnatural\partightenfactor0

\f0\fs36 \cf2 \CocoaLigature0 \
\cf3 typedef\cf2  \cf3 struct\cf2  Data\
\{\
    double event_time; \cf4 // Min-heap ordered by this variable\cf2 \
    int subvolume;\
\} data;\
\
\cf3 typedef\cf2  \cf3 struct\cf2  Heap\
\{\
    int current_number_of_elements_in_array; \cf4 // Default 0\cf2 \
    \cf3 struct\cf2  Data data [\cf5 100\cf2 ];\
\} heap;\
\
int get_left_child_index (int current_position);\
\
int get_right_child_index (int current_position);\
\
int get_parent_index (int current_position);\
\
\
\cf3 struct\cf2  Data get_left_child (int index, \cf3 struct\cf2  Data data []);\
\
\cf3 struct\cf2  Data get_right_child (int index, \cf3 struct\cf2  Data data []);\
\
\cf3 struct\cf2  Data get_parent (int index, \cf3 struct\cf2  Data data[]);\
\
bool has_left_child (int index, int current_number_of_elements_in_array);\
\
bool has_right_child (int index, int current_number_of_elements_in_array);\
\
bool has_parent (int index, int current_number_of_elements_in_array);\
\
\
\cf3 void\cf2  add (\cf3 struct\cf2  Data element, \cf3 struct\cf2  Heap *heap);\
\
\cf3 void\cf2  perlocate_up (\cf3 struct\cf2  Data data [], int current_number_of_elements);\
\
\cf3 void\cf2  remove_element (\cf3 struct\cf2  Heap *heap);\
\
\cf3 void\cf2  perlocate_down (\cf3 struct\cf2  Data data [], int current_number_of_elements);\
\
\cf3 void\cf2  swap (\cf3 struct\cf2  Data data [], int indexOne, int indexTwo);\
\
\cf3 void\cf2  print_heap_array (\cf3 struct\cf2  Heap heap);\
\
\
\
\
\
\
int get_left_child_index (int current_position)\
\{\
    return (current_position*\cf5 2\cf2 )+\cf5 1\cf2 ;\
\}\
\
int get_right_child_index (int current_position)\
\{\
    return (current_position*\cf5 2\cf2 )+\cf5 2\cf2 ;\
\}\
\
int get_parent_index (int current_position)\
\{\
    return (current_position-\cf5 1\cf2 )/\cf5 2\cf2 ;\
\}\
\
\
\cf3 struct\cf2  Data get_left_child (int index, \cf3 struct\cf2  Data data [])\
\{\
    return data[get_left_child_index(index)];\
\}\
\
\cf3 struct\cf2  Data get_right_child (int index, \cf3 struct\cf2  Data data [])\
\{\
    return data[get_right_child_index(index)];\
\}\
\
\cf3 struct\cf2  Data get_parent (int index, \cf3 struct\cf2  Data data [])\
\{\
    return data[get_parent_index(index)];\
\}\
\
bool has_left_child (int index, int current_number_of_elements_in_array)\
\{\
    return get_left_child_index(index) < current_number_of_elements_in_array;\
\}\
\
bool has_right_child (int index, int current_number_of_elements_in_array)\
\{\
    return get_right_child_index(index) < current_number_of_elements_in_array;\
\}\
\
bool has_parent (int index, int current_number_of_elements_in_array)\
\{\
    return get_parent_index(index) < current_number_of_elements_in_array;\
\}\
\
\cf3 void\cf2  add (\cf3 struct\cf2  Data element, \cf3 struct\cf2  Heap *heap)\
\{\
    \cf4 // The array does not resize itself\cf2 \
   \
    heap->data[heap->current_number_of_elements_in_array] = element;\
    heap->current_number_of_elements_in_array += \cf5 1\cf2 ;\
\
    perlocate_up(heap->data, heap->current_number_of_elements_in_array);\
\}\
\
\cf3 void\cf2  perlocate_up (\cf3 struct\cf2  Data data [], int current_number_of_elements)\
\{\
    double event_time = data[current_number_of_elements-\cf5 1\cf2 ].event_time;\
    int index = current_number_of_elements-\cf5 1\cf2 ;\
    \
    while (has_parent(index, current_number_of_elements) && event_time < get_parent(index, data).event_time)\
    \{\
        swap(data, index,get_parent_index(index));\
        index = get_parent_index(index);\
    \}\
    \
\}\
\
\cf3 void\cf2  remove_element (\cf3 struct\cf2  Heap *heap)\
\{\
    if (heap->current_number_of_elements_in_array == \cf5 0\cf2 )\
    \{\
        return;\
    \}\
    \
    heap->data[\cf5 0\cf2 ] = heap->data[heap->current_number_of_elements_in_array-\cf5 1\cf2 ];\
    \
    \cf4 // Dummy value\cf2 \
    \cf3 struct\cf2  Data new_data;\
    new_data.event_time = -\cf5 1\cf2 ;\
    new_data.subvolume = -\cf5 1\cf2 ;\
    \
    heap->data[heap->current_number_of_elements_in_array-\cf5 1\cf2 ] = new_data;\
    heap->current_number_of_elements_in_array-=\cf5 1\cf2 ;\
    \
    perlocate_down(heap->data, heap->current_number_of_elements_in_array);\
\}\
\
\cf3 void\cf2  perlocate_down (\cf3 struct\cf2  Data data [], int current_number_of_elements)\
\{\
    double event_time = data[\cf5 0\cf2 ].event_time;\
    int index = \cf5 0\cf2 ;\
    \
    while (has_left_child(index, current_number_of_elements))\
    \{\
        int smaller_index = get_left_child_index(index);\
        \
        if (has_right_child(index, current_number_of_elements) && get_right_child(index, data).event_time < get_left_child(index, data).event_time)\
        \{\
            smaller_index = get_right_child_index(index);\
        \}\
        \
        if (event_time < data[smaller_index].event_time)\
        \{\
            break;\
        \} else\
        \{\
            swap(data, smaller_index, index);\
        \}\
        index = smaller_index;\
    \}\
\}\
           \
\cf3 void\cf2  swap (\cf3 struct\cf2  Data data [], int indexOne, int indexTwo)\
\{\
    \cf3 struct\cf2  Data temp = data[indexOne];\
    data[indexOne] = data[indexTwo];\
    data[indexTwo] = temp;\
\}\
\
\cf3 void\cf2  print_heap_array (\cf3 struct\cf2  Heap heap)\
\{\
    for (int i = \cf5 0\cf2  ; i < heap.current_number_of_elements_in_array ; i++)\
    \{\
        printf(\cf6 "%f"\cf2 , heap.data[i].event_time);\
        printf(\cf6 "---"\cf2 );\
        printf(\cf6 "%d"\cf2 , heap.data[i].subvolume);\
        printf(\cf6 "\\n"\cf2 );\
    \}\
\}}