#ifdef K_DOUBLE_PRECISION
#define DOUBLE_PRECISION
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#elif AMD_DOUBLE_PRECISION
#define DOUBLE_PRECISION
#pragma OPENCL EXTENSION cl_amd_fp64: enable
#endif

// Macros to explicitly control precision of the constants, otherwise
// known to cause problems for some Compilers
#ifdef DOUBLE_PRECISION
#define CPREC(a) a
#else
#define CPREC(a) a##f
#endif

//replace divisions by multiplication with the reciprocal
#define REPLACE_DIV_WITH_RCP 1

//Call the appropriate math function based on precision
#ifdef DOUBLE_PRECISION
#define real double
#if REPLACE_DIV_WITH_RCP
#define DIV(x,y) ((x)*(1.0/(y)))
#else
#define DIV(x,y) ((x)/(y))
#endif
#define POW pow
#define EXP exp
#define EXP10 exp10
#define EXP2 exp2
#define MAX fmax
#define MIN fmin
#define LOG log
#define LOG10 log10
#else
#define real float
#if REPLACE_DIV_WITH_RCP
#define DIV(x,y) ((x)*(1.0f/(y)))
#else
#define DIV(x,y) ((x)/(y))
#endif
#define POW pow
#define EXP exp
#define EXP10 exp10
#define EXP2 exp2
#define MAX fmax
#define MIN fmin
#define LOG log
#define LOG10 log10
#endif
//Kernel indexing macros
#define thread_num (get_global_id(0))
#define idx2(p,z) (p[(((z)-1)*(N_GP)) + thread_num])
#define idx(x, y) ((x)[(y)-1])
#define C(q)     idx2(C, q)
#define Y(q)     idx2(Y, q)
#define RF(q)    idx2(RF, q)
#define EG(q)    idx2(EG, q)
#define RB(q)    idx2(RB, q)
#define RKLOW(q) idx2(RKLOW, q)
#define ROP(q)   idx(ROP, q)
#define WDOT(q)  idx2(WDOT, q)
#define RKF(q)   idx2(RKF, q)
#define RKR(q)   idx2(RKR, q)
#define A_DIM    (11)
#define A(b, c)  idx2(A, (((b)*A_DIM)+c) )


__kernel void
ratx2_kernel(__global const real* C, __global real* RF)
{

    RF(1) = RF(1)*C(2)*C(4);
    RF(2) = RF(2)*C(3)*C(1);
    RF(3) = RF(3)*C(5)*C(1);
    RF(4) = RF(4)*C(5)*C(5);
    RF(6) = RF(6)*C(2)*C(2)*C(1);
    RF(7) = RF(7)*C(2)*C(2)*C(6);
    RF(8) = RF(8)*C(2)*C(2)*C(12);
    RF(13) = RF(13)*C(2)*C(4)*C(4);
    RF(14) = RF(14)*C(2)*C(4)*C(6);
    RF(15) = RF(15)*C(2)*C(4)*C(22);
    RF(16) = RF(16)*C(5)*C(5);
    RF(17) = RF(17)*C(7)*C(2);
    RF(18) = RF(18)*C(7)*C(2);
    RF(19) = RF(19)*C(7)*C(2);
    RF(20) = RF(20)*C(7)*C(3);
    RF(21) = RF(21)*C(7)*C(5);
    RF(22) = RF(22)*C(7)*C(7);
    RF(23) = RF(23)*C(7)*C(7);
    RF(24) = RF(24)*C(8)*C(2);
    RF(25) = RF(25)*C(8)*C(2);
    RF(26) = RF(26)*C(8)*C(3);
    RF(27) = RF(27)*C(8)*C(5);
    RF(28) = RF(28)*C(8)*C(5);
    RF(30) = RF(30)*C(11)*C(5);
    RF(31) = RF(31)*C(11)*C(1);
    RF(32) = RF(32)*C(11)*C(4);
    RF(33) = RF(33)*C(11)*C(7);
    RF(34) = RF(34)*C(3);
    RF(35) = RF(35)*C(5);
    RF(36) = RF(36)*C(1);
    RF(37) = RF(37)*C(6);
    RF(38) = RF(38)*C(4);
    RF(39) = RF(39)*C(11);
    RF(40) = RF(40)*C(12);
    RF(41) = RF(41)*C(2);
    RF(42) = RF(42)*C(2);
    RF(43) = RF(43)*C(3);
    RF(44) = RF(44)*C(3);
    RF(45) = RF(45)*C(5);
    RF(47) = RF(47)*C(4);
    RF(48) = RF(48)*C(2);
    RF(49) = RF(49)*C(1);
    RF(50) = RF(50)*C(3);
    RF(51) = RF(51)*C(4);
    RF(52) = RF(52)*C(4);
    RF(53) = RF(53)*C(5);
    RF(54) = RF(54)*C(5);
    RF(55) = RF(55)*C(7);
    RF(56) = RF(56)*C(11);
    RF(59) = RF(59)*C(22);
    RF(60) = RF(60)*C(2);
    RF(61) = RF(61)*C(3);
    RF(62) = RF(62)*C(3);
    RF(63) = RF(63)*C(5);
    RF(64) = RF(64)*C(1);
    RF(65) = RF(65)*C(4);
    RF(66) = RF(66)*C(4);
    RF(67) = RF(67)*C(6);
    RF(68) = RF(68)*C(11);
    RF(69) = RF(69)*C(12);
    RF(70) = RF(70)*C(12);
    RF(71) = RF(71)*C(13)*C(2);
    RF(72) = RF(72)*C(13)*C(2);
    RF(73) = RF(73)*C(13)*C(3);
    RF(74) = RF(74)*C(13)*C(5);
    RF(75) = RF(75)*C(13)*C(4);
    RF(76) = RF(76)*C(13)*C(7);
    RF(77) = RF(77)*C(13);
    RF(78) = RF(78)*C(9)*C(2);
    RF(79) = RF(79)*C(9)*C(3);
    RF(80) = RF(80)*C(9)*C(5);
    RF(81) = RF(81)*C(9)*C(5);
    RF(82) = RF(82)*C(9)*C(4);
    RF(83) = RF(83)*C(9)*C(4);
    RF(84) = RF(84)*C(9)*C(7);
    RF(85) = RF(85)*C(9)*C(7);
    RF(86) = RF(86)*C(9)*C(8);
    RF(87) = RF(87)*C(9);
    RF(88) = RF(88)*C(9);
    RF(89) = RF(89)*C(9);
    RF(90) = RF(90)*C(9)*C(13);
    RF(91) = RF(91)*C(9);
    RF(92) = RF(92)*C(9);
    RF(93) = RF(93)*C(9)*C(9);
    RF(94) = RF(94)*C(9)*C(9);
    RF(95) = RF(95)*C(9)*C(17);
    RF(96) = RF(96)*C(2);
    RF(97) = RF(97)*C(2);
    RF(98) = RF(98)*C(2);
    RF(99) = RF(99)*C(3);
    RF(100) = RF(100)*C(5);
    RF(101) = RF(101)*C(4);
    RF(102) = RF(102)*C(10)*C(2);
    RF(103) = RF(103)*C(10)*C(3);
    RF(104) = RF(104)*C(10)*C(5);
    RF(105) = RF(105)*C(10);
    RF(106) = RF(106)*C(10);
    RF(107) = RF(107)*C(10);
    RF(108) = RF(108)*C(17)*C(2);
    RF(109) = RF(109)*C(17)*C(3);
    RF(110) = RF(110)*C(17)*C(4);
    RF(111) = RF(111)*C(17);
    RF(112) = RF(112)*C(17);
    RF(113) = RF(113)*C(17)*C(17);
    RF(114) = RF(114)*C(14);
    RF(116) = RF(116)*C(14)*C(3);
    RF(117) = RF(117)*C(14)*C(3);
    RF(118) = RF(118)*C(14)*C(5);
    RF(119) = RF(119)*C(14)*C(5);
    RF(120) = RF(120)*C(14);
    RF(122) = RF(122)*C(2);
    RF(123) = RF(123)*C(3);
    RF(124) = RF(124)*C(5);
    RF(125) = RF(125)*C(4);
    RF(126) = RF(126)*C(18)*C(2);
    RF(127) = RF(127)*C(18)*C(2);
    RF(128) = RF(128)*C(18)*C(2);
    RF(129) = RF(129)*C(18)*C(3);
    RF(130) = RF(130)*C(18)*C(3);
    RF(131) = RF(131)*C(18)*C(5);
    RF(132) = RF(132)*C(2);
    RF(133) = RF(133)*C(2);
    RF(134) = RF(134)*C(2);
    RF(135) = RF(135)*C(3);
    RF(136) = RF(136)*C(3);
    RF(137) = RF(137)*C(5);
    RF(138) = RF(138)*C(4);
    RF(139) = RF(139)*C(4);
    RF(140) = RF(140)*C(4);
    RF(141) = RF(141)*C(7);
    RF(142) = RF(142)*C(8);
    RF(144) = RF(144)*C(9);
    RF(145) = RF(145)*C(9);
    RF(146) = RF(146)*C(9);
    RF(148) = RF(148)*C(2);
    RF(149) = RF(149)*C(2);
    RF(150) = RF(150)*C(2);
    RF(151) = RF(151)*C(3);
    RF(152) = RF(152)*C(5);
    RF(153) = RF(153)*C(4);
    RF(154) = RF(154)*C(4);
    RF(155) = RF(155)*C(15);
    RF(156) = RF(156)*C(15)*C(2);
    RF(157) = RF(157)*C(15)*C(2);
    RF(158) = RF(158)*C(15)*C(3);
    RF(159) = RF(159)*C(15)*C(3);
    RF(160) = RF(160)*C(15)*C(3);
    RF(161) = RF(161)*C(15)*C(5);
    RF(162) = RF(162)*C(15)*C(4);
    RF(163) = RF(163)*C(15)*C(7);
    RF(164) = RF(164)*C(15);
    RF(165) = RF(165)*C(15);
    RF(166) = RF(166)*C(15);
    RF(167) = RF(167)*C(15);
    RF(168) = RF(168)*C(15)*C(9);
    RF(169) = RF(169)*C(15)*C(9);
    RF(170) = RF(170)*C(2);
    RF(171) = RF(171)*C(2);
    RF(172) = RF(172)*C(3);
    RF(173) = RF(173)*C(3);
    RF(174) = RF(174)*C(4);
    RF(175) = RF(175)*C(7);
    RF(176) = RF(176)*C(7);
    RF(177) = RF(177)*C(7);
    RF(178) = RF(178)*C(8);
    RF(180) = RF(180)*C(16)*C(2);
    RF(181) = RF(181)*C(16)*C(3);
    RF(182) = RF(182)*C(16)*C(5);
    RF(183) = RF(183)*C(16);
    RF(184) = RF(184)*C(16)*C(9);
    RF(185) = RF(185)*C(20)*C(2);
    RF(186) = RF(186)*C(20)*C(2);
    RF(187) = RF(187)*C(20)*C(7);
    RF(188) = RF(188)*C(20)*C(7);
    RF(189) = RF(189)*C(20);
    RF(190) = RF(190)*C(21)*C(2);
    RF(191) = RF(191)*C(21)*C(2);
    RF(192) = RF(192)*C(21)*C(2);
    RF(193) = RF(193)*C(21)*C(3);
    RF(194) = RF(194)*C(21)*C(3);
    RF(195) = RF(195)*C(21)*C(3);
    RF(196) = RF(196)*C(21)*C(5);
    RF(197) = RF(197)*C(21)*C(7);
    RF(198) = RF(198)*C(21)*C(9);
    RF(199) = RF(199)*C(2);
    RF(200) = RF(200)*C(2);
    RF(201) = RF(201)*C(3);
    RF(202) = RF(202)*C(5);
    RF(203) = RF(203)*C(4);
    RF(204) = RF(204)*C(7);
    RF(205) = RF(205)*C(9);
}
