FUNCTION Z_STRING_XOR.
*"----------------------------------------------------------------------
*"*"Local Interface:
*"  IMPORTING
*"     VALUE(IV_A) TYPE  STRING
*"     VALUE(IV_B) TYPE  STRING
*"  EXPORTING
*"     VALUE(RV_RESULT) TYPE  STRING
*"----------------------------------------------------------------------

  DATA: lv_result TYPE string.

  DATA: lv_length_a TYPE i,
        lv_length_b TYPE i,
        lv_index type i.

  lv_length_a = strlen( iv_a ).
  lv_length_b = strlen( iv_b ).

  IF lv_length_a <> lv_length_b.
    MESSAGE 'Both input strings must have the same length.' TYPE 'E'.
    RETURN.
  ENDIF.

  lv_index = 0.

  DO lv_length_a TIMES.
    if iv_a+lv_index(1) = iv_b+lv_index(1).
      CONCATENATE RV_RESULT '0' INTO RV_RESULT.
    ELSE.
      CONCATENATE RV_RESULT '1' INTO RV_RESULT.
    ENDIF.
    lv_index = lv_index + 1.
  ENDDO.

ENDFUNCTION.