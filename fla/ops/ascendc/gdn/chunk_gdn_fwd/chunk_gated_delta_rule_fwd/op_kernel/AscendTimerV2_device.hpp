// Shared GDN timer; supports source-tree and flattened OPP headers.
#if __has_include("../../../../common/kernel_utils/gdn_timer/AscendTimerV2_device.hpp")
#include "../../../../common/kernel_utils/gdn_timer/AscendTimerV2_device.hpp"
#else
#include "kernel_utils/gdn_timer/AscendTimerV2_device.hpp"
#endif
