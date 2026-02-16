test_frsqrtex_non_ieee_cases_1:
  #_ REGISTER_IN f1 0x0000000000000001
  mtfsfi 7, 0
  frsqrte f3, f1
  blr
  #_ REGISTER_OUT f1 0x0000000000000001
  #_ REGISTER_OUT f3 0x617F100000000000

test_frsqrtex_non_ieee_cases_2:
  #_ REGISTER_IN f1 0x0000000000000001
  mtfsfi 7, 4
  frsqrte f3, f1
  blr
  #_ REGISTER_OUT f1 0x0000000000000001
  #_ REGISTER_OUT f3 0x7FF0000000000000

test_frsqrtex_non_ieee_cases_3:
  #_ REGISTER_IN f1 0x8000000000000001
  mtfsfi 7, 0
  frsqrte f3, f1
  blr
  #_ REGISTER_OUT f1 0x8000000000000001
  #_ REGISTER_OUT f3 0x7FF8000000000000

test_frsqrtex_non_ieee_cases_4:
  #_ REGISTER_IN f1 0x8000000000000001
  mtfsfi 7, 4
  frsqrte f3, f1
  blr
  #_ REGISTER_OUT f1 0x8000000000000001
  #_ REGISTER_OUT f3 0xFFF0000000000000
