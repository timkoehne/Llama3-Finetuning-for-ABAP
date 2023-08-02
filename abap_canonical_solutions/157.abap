FUNCTION z_right_angle_triangle.
*"----------------------------------------------------------------------
*"*"Local Interface:
*"  IMPORTING
*"     VALUE(A) TYPE  I
*"     VALUE(B) TYPE  I
*"     VALUE(C) TYPE  I
*"  EXPORTING
*"     VALUE(RETURN_VALUE) TYPE  C
*"----------------------------------------------------------------------

  DATA: square_a TYPE i,
        square_b TYPE i,
        square_c TYPE i.

  square_a = a * a.
  square_b = b * b.
  square_c = c * c.

  return_value = ' '.
  if square_a = square_b + square_c or square_b = square_a + square_c or square_c = square_a + square_b.
    return_value = 'X'.
  endif.

ENDFUNCTION.