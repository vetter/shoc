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
ratx4_kernel(__global const real* C, __global real* RB)
{

    RB(1) = RB(1)*C(3)*C(5);
    RB(2) = RB(2)*C(2)*C(5);
    RB(3) = RB(3)*C(2)*C(6);
    RB(4) = RB(4)*C(3)*C(6);
    RB(6) = RB(6)*C(1)*C(1);
    RB(7) = RB(7)*C(1)*C(6);
    RB(8) = RB(8)*C(1)*C(12);
    RB(13) = RB(13)*C(7)*C(4);
    RB(14) = RB(14)*C(7)*C(6);
    RB(15) = RB(15)*C(7)*C(22);
    RB(16) = RB(16)*C(8);
    RB(17) = RB(17)*C(3)*C(6);
    RB(18) = RB(18)*C(4)*C(1);
    RB(19) = RB(19)*C(5)*C(5);
    RB(20) = RB(20)*C(5)*C(4);
    RB(21) = RB(21)*C(4)*C(6);
    RB(22) = RB(22)*C(4)*C(8);
    RB(23) = RB(23)*C(4)*C(8);
    RB(24) = RB(24)*C(7)*C(1);
    RB(25) = RB(25)*C(5)*C(6);
    RB(26) = RB(26)*C(5)*C(7);
    RB(27) = RB(27)*C(7)*C(6);
    RB(28) = RB(28)*C(7)*C(6);
    RB(30) = RB(30)*C(12)*C(2);
    RB(31) = RB(31)*C(13);
    RB(32) = RB(32)*C(12)*C(3);
    RB(33) = RB(33)*C(12)*C(5);
    RB(34) = RB(34)*C(11)*C(2);
    RB(35) = RB(35)*C(2);
    RB(36) = RB(36)*C(2);
    RB(37) = RB(37)*C(13)*C(2);
    RB(38) = RB(38)*C(3);
    RB(39) = RB(39)*C(17);
    RB(40) = RB(40)*C(11);
    RB(41) = RB(41)*C(13);
    RB(42) = RB(42)*C(11)*C(1);
    RB(43) = RB(43)*C(11)*C(5);
    RB(44) = RB(44)*C(12)*C(2);
    RB(45) = RB(45)*C(11)*C(6);
    RB(47) = RB(47)*C(11)*C(7);
    RB(48) = RB(48)*C(9);
    RB(49) = RB(49)*C(2)*C(9);
    RB(50) = RB(50)*C(2);
    RB(51) = RB(51)*C(5);
    RB(52) = RB(52)*C(12)*C(2)*C(2);
    RB(53) = RB(53)*C(13)*C(2);
    RB(54) = RB(54)*C(6);
    RB(55) = RB(55)*C(13)*C(5);
    RB(56) = RB(56)*C(18);
    RB(57) = RB(57)*C(14)*C(2);
    RB(58) = RB(58)*C(14)*C(1);
    RB(59) = RB(59)*C(22);
    RB(60) = RB(60)*C(1);
    RB(61) = RB(61)*C(11)*C(1);
    RB(62) = RB(62)*C(2);
    RB(63) = RB(63)*C(13)*C(2);
    RB(64) = RB(64)*C(9)*C(2);
    RB(65) = RB(65)*C(2)*C(5)*C(11);
    RB(66) = RB(66)*C(11)*C(6);
    RB(67) = RB(67)*C(6);
    RB(68) = RB(68)*C(11);
    RB(69) = RB(69)*C(12);
    RB(70) = RB(70)*C(13)*C(11);
    RB(72) = RB(72)*C(1);
    RB(73) = RB(73)*C(5);
    RB(74) = RB(74)*C(6);
    RB(75) = RB(75)*C(7);
    RB(76) = RB(76)*C(8);
    RB(77) = RB(77)*C(18)*C(2);
    RB(78) = RB(78)*C(10);
    RB(79) = RB(79)*C(13)*C(2);
    RB(80) = RB(80)*C(6);
    RB(81) = RB(81)*C(6);
    RB(82) = RB(82)*C(3);
    RB(83) = RB(83)*C(5)*C(13);
    RB(84) = RB(84)*C(10)*C(4);
    RB(85) = RB(85)*C(5);
    RB(86) = RB(86)*C(10)*C(7);
    RB(87) = RB(87)*C(2);
    RB(88) = RB(88)*C(10)*C(11);
    RB(89) = RB(89)*C(19);
    RB(90) = RB(90)*C(10);
    RB(91) = RB(91)*C(15)*C(2);
    RB(92) = RB(92)*C(15)*C(2);
    RB(93) = RB(93)*C(16);
    RB(94) = RB(94)*C(2);
    RB(95) = RB(95)*C(15)*C(11);
    RB(96) = RB(96)*C(13)*C(1);
    RB(97) = RB(97)*C(9)*C(5);
    RB(98) = RB(98)*C(6);
    RB(99) = RB(99)*C(13)*C(5);
    RB(100) = RB(100)*C(13)*C(6);
    RB(101) = RB(101)*C(13)*C(7);
    RB(102) = RB(102)*C(9)*C(1);
    RB(103) = RB(103)*C(9)*C(5);
    RB(104) = RB(104)*C(9)*C(6);
    RB(105) = RB(105)*C(15)*C(2);
    RB(106) = RB(106)*C(9)*C(9);
    RB(107) = RB(107)*C(9)*C(9);
    RB(108) = RB(108)*C(11);
    RB(109) = RB(109)*C(2)*C(11)*C(11);
    RB(110) = RB(110)*C(5)*C(11)*C(11);
    RB(111) = RB(111)*C(14)*C(11);
    RB(112) = RB(112)*C(11);
    RB(113) = RB(113)*C(14)*C(11)*C(11);
    RB(115) = RB(115)*C(14)*C(2);
    RB(116) = RB(116)*C(17)*C(2);
    RB(117) = RB(117)*C(11);
    RB(118) = RB(118)*C(18)*C(2);
    RB(119) = RB(119)*C(9)*C(11);
    RB(120) = RB(120)*C(11);
    RB(122) = RB(122)*C(14)*C(2);
    RB(123) = RB(123)*C(11);
    RB(124) = RB(124)*C(18)*C(2);
    RB(125) = RB(125)*C(12);
    RB(127) = RB(127)*C(17)*C(1);
    RB(128) = RB(128)*C(9)*C(11);
    RB(129) = RB(129)*C(17)*C(5);
    RB(130) = RB(130)*C(12);
    RB(131) = RB(131)*C(17)*C(6);
    RB(132) = RB(132)*C(15);
    RB(133) = RB(133)*C(14)*C(1);
    RB(134) = RB(134)*C(1);
    RB(135) = RB(135)*C(18)*C(2);
    RB(136) = RB(136)*C(9)*C(11);
    RB(137) = RB(137)*C(14)*C(6);
    RB(138) = RB(138)*C(14)*C(7);
    RB(139) = RB(139)*C(3);
    RB(140) = RB(140)*C(13);
    RB(141) = RB(141)*C(5);
    RB(142) = RB(142)*C(15)*C(7);
    RB(143) = RB(143)*C(15)*C(11);
    RB(144) = RB(144)*C(14)*C(10);
    RB(145) = RB(145)*C(21);
    RB(146) = RB(146)*C(20)*C(2);
    RB(147) = RB(147)*C(9)*C(11);
    RB(148) = RB(148)*C(19);
    RB(149) = RB(149)*C(9);
    RB(150) = RB(150)*C(18)*C(1);
    RB(151) = RB(151)*C(18)*C(5);
    RB(152) = RB(152)*C(18)*C(6);
    RB(153) = RB(153)*C(18)*C(7);
    RB(154) = RB(154)*C(13)*C(11)*C(5);
    RB(155) = RB(155)*C(1);
    RB(157) = RB(157)*C(1);
    RB(158) = RB(158)*C(5);
    RB(159) = RB(159)*C(9);
    RB(160) = RB(160)*C(13);
    RB(161) = RB(161)*C(6);
    RB(162) = RB(162)*C(7);
    RB(163) = RB(163)*C(19)*C(5);
    RB(164) = RB(164)*C(11);
    RB(165) = RB(165)*C(20)*C(2);
    RB(166) = RB(166)*C(10);
    RB(167) = RB(167)*C(20)*C(2);
    RB(168) = RB(168)*C(10);
    RB(170) = RB(170)*C(16);
    RB(171) = RB(171)*C(15)*C(1);
    RB(172) = RB(172)*C(9)*C(13);
    RB(173) = RB(173)*C(19)*C(2);
    RB(174) = RB(174)*C(15)*C(7);
    RB(175) = RB(175)*C(16)*C(4);
    RB(176) = RB(176)*C(15)*C(8);
    RB(177) = RB(177)*C(9)*C(13)*C(5);
    RB(178) = RB(178)*C(16)*C(7);
    RB(179) = RB(179)*C(16)*C(11);
    RB(180) = RB(180)*C(1);
    RB(181) = RB(181)*C(5);
    RB(182) = RB(182)*C(6);
    RB(183) = RB(183)*C(9);
    RB(184) = RB(184)*C(10);
    RB(185) = RB(185)*C(21);
    RB(186) = RB(186)*C(10);
    RB(187) = RB(187)*C(21)*C(4);
    RB(188) = RB(188)*C(5)*C(13);
    RB(189) = RB(189)*C(21)*C(11);
    RB(191) = RB(191)*C(15)*C(9);
    RB(192) = RB(192)*C(20)*C(1);
    RB(193) = RB(193)*C(18)*C(9)*C(2);
    RB(195) = RB(195)*C(20)*C(5);
    RB(196) = RB(196)*C(20)*C(6);
    RB(197) = RB(197)*C(20)*C(8);
    RB(198) = RB(198)*C(20)*C(10);
    RB(199) = RB(199)*C(9);
    RB(200) = RB(200)*C(21)*C(1);
    RB(201) = RB(201)*C(13);
    RB(202) = RB(202)*C(21)*C(6);
    RB(203) = RB(203)*C(21)*C(7);
    RB(204) = RB(204)*C(5)*C(13);
    RB(205) = RB(205)*C(10)*C(21);
    RB(206) = RB(206)*C(20)*C(9);
}
