// Shared GDN timer; supports source-tree and flattened OPP headers.
#if __has_include("../../../../common/kernel_utils/gdn_timer/AscendTimerV2.hpp")
#include "../../../../common/kernel_utils/gdn_timer/AscendTimerV2.hpp"
#else
#include "kernel_utils/gdn_timer/AscendTimerV2.hpp"
#endif
