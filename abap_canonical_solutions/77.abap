FUNCTION Z_ISCUBE.
*"----------------------------------------------------------------------
*"*"Local Interface:
*"  IMPORTING
*"     VALUE(A) TYPE  I
*"  EXPORTING
*"     VALUE(RETURN_VALUE) TYPE  C
*"----------------------------------------------------------------------

  DATA: abs_a    TYPE i,
        root_a   TYPE i,
        cube_a   TYPE i.

  abs_a = ABS( a ).
  root_a = abs_a ** ( 1 / 3 ).
  cube_a = root_a ** 3.

  if cube_a = abs_a.
    return_value = 'X'.
  else.
    return_value = ' '.
  ENDIF.


ENDFUNCTION.