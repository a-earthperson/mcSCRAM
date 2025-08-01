// Auto-generated file - DO NOT EDIT
// Generated from ./share/report.rng

#include <string_view>

namespace scram::schemas {

constexpr std::string_view REPORT_SCHEMA = 
"<grammar xmlns=\"http://relaxng.org/ns/structure/1.0\"\n"
"         datatypeLibrary=\"http://www.w3.org/2001/XMLSchema-datatypes\">\n"
"\n"
"    <!-- ############################################################### -->\n"
"    <!-- Report Layer -->\n"
"    <!-- ############################################################### -->\n"
"\n"
"    <start>\n"
"        <element name=\"report\">\n"
"            <ref name=\"information-layer\"/>\n"
"            <optional>\n"
"                <ref name=\"results-layer\"/>\n"
"            </optional>\n"
"        </element>\n"
"    </start>\n"
"\n"
"    <define name=\"probability-data\">  <!-- [0.0, 1.0] values for probability -->\n"
"        <data type=\"double\">\n"
"            <param name=\"minInclusive\">0</param>\n"
"            <param name=\"maxInclusive\">1</param>\n"
"        </data>\n"
"    </define>\n"
"\n"
"    <!-- ############################################################### -->\n"
"    <!-- I. Information Layer -->\n"
"    <!-- ############################################################### -->\n"
"\n"
"    <define name=\"information-layer\">\n"
"        <element name=\"information\">\n"
"            <element name=\"software\">\n"
"                <attribute name=\"name\"> <data type=\"string\"/> </attribute>\n"
"                <attribute name=\"version\"> <data type=\"string\"/> </attribute>\n"
"                <optional>\n"
"                    <attribute name=\"contacts\"> <text/> </attribute>\n"
"                </optional>\n"
"            </element>\n"
"            <element name=\"time\"> <data type=\"string\"/> </element>\n"
"            <optional>\n"
"                <ref name=\"performance-info\"/>\n"
"            </optional>\n"
"            <zeroOrMore>\n"
"                <ref name=\"calculated-quantity\"/>\n"
"            </zeroOrMore>\n"
"            <element name=\"model-features\">\n"
"                <optional>\n"
"                    <attribute name=\"name\"> <data type=\"NCName\"/> </attribute>\n"
"                </optional>\n"
"                <optional>\n"
"                    <element name=\"gates\">\n"
"                        <data type=\"nonNegativeInteger\"/>\n"
"                    </element>\n"
"                </optional>\n"
"                <optional>\n"
"                    <element name=\"basic-events\">\n"
"                        <data type=\"nonNegativeInteger\"/>\n"
"                    </element>\n"
"                </optional>\n"
"                <optional>\n"
"                    <element name=\"house-events\">\n"
"                        <data type=\"nonNegativeInteger\"/>\n"
"                    </element>\n"
"                </optional>\n"
"                <optional>\n"
"                    <element name=\"ccf-groups\">\n"
"                        <data type=\"nonNegativeInteger\"/>\n"
"                    </element>\n"
"                </optional>\n"
"                <optional>\n"
"                    <element name=\"fault-trees\">\n"
"                        <data type=\"nonNegativeInteger\"/>\n"
"                    </element>\n"
"                </optional>\n"
"                <optional>\n"
"                    <element name=\"event-trees\">\n"
"                        <data type=\"nonNegativeInteger\"/>\n"
"                    </element>\n"
"                </optional>\n"
"                <optional>\n"
"                    <element name=\"functional-events\">\n"
"                        <data type=\"nonNegativeInteger\"/>\n"
"                    </element>\n"
"                </optional>\n"
"                <optional>\n"
"                    <element name=\"sequences\">\n"
"                        <data type=\"nonNegativeInteger\"/>\n"
"                    </element>\n"
"                </optional>\n"
"                <optional>\n"
"                    <element name=\"rules\">\n"
"                        <data type=\"nonNegativeInteger\"/>\n"
"                    </element>\n"
"                </optional>\n"
"                <optional>\n"
"                    <element name=\"initiating-events\">\n"
"                        <data type=\"nonNegativeInteger\"/>\n"
"                    </element>\n"
"                </optional>\n"
"            </element>\n"
"            <zeroOrMore>\n"
"                <element name=\"warning\"> <text/> </element>\n"
"            </zeroOrMore>\n"
"            <optional>\n"
"                <element name=\"feedback\"> <text/> </element>\n"
"            </optional>\n"
"        </element>\n"
"    </define>\n"
"\n"
"    <define name=\"analysis-id\">\n"
"        <attribute name=\"name\"> <data type=\"NCName\"/> </attribute>\n"
"        <optional>\n"
"            <attribute name=\"initiating-event\"> <data type=\"NCName\"/> </attribute>\n"
"        </optional>\n"
"        <optional>\n"
"            <group>\n"
"                <attribute name=\"alignment\"> <data type=\"NCName\"/> </attribute>\n"
"                <attribute name=\"phase\"> <data type=\"NCName\"/> </attribute>\n"
"            </group>\n"
"        </optional>\n"
"        <optional>\n"
"            <attribute name=\"description\"> <text/> </attribute>\n"
"        </optional>\n"
"        <optional>\n"
"            <attribute name=\"warning\"> <text/> </attribute>\n"
"        </optional>\n"
"    </define>\n"
"\n"
"    <define name=\"performance-info\">\n"
"        <element name=\"performance\">\n"
"            <oneOrMore>\n"
"                <element name=\"calculation-time\">\n"
"                    <ref name=\"analysis-id\"/>\n"
"                    <optional>\n"
"                        <element name=\"products\">\n"
"                            <data type=\"double\"/>\n"
"                        </element>\n"
"                    </optional>\n"
"                    <optional>\n"
"                        <element name=\"probability\">\n"
"                            <data type=\"double\"/>\n"
"                        </element>\n"
"                    </optional>\n"
"                    <optional>\n"
"                        <element name=\"importance\">\n"
"                            <data type=\"double\"/>\n"
"                        </element>\n"
"                    </optional>\n"
"                    <optional>\n"
"                        <element name=\"uncertainty\">\n"
"                            <data type=\"double\"/>\n"
"                        </element>\n"
"                    </optional>\n"
"                </element>\n"
"            </oneOrMore>\n"
"        </element>\n"
"    </define>\n"
"\n"
"    <define name=\"calculated-quantity\">\n"
"        <element name=\"calculated-quantity\">\n"
"            <attribute name=\"name\"> <text/> </attribute>\n"
"            <optional>\n"
"                <attribute name=\"definition\"> <text/> </attribute>\n"
"            </optional>\n"
"            <zeroOrMore>\n"
"                <attribute name=\"approximation\"> <text/> </attribute>\n"
"            </zeroOrMore>\n"
"            <optional>\n"
"                <ref name=\"calculation-method\"/>\n"
"            </optional>\n"
"        </element>\n"
"    </define>\n"
"\n"
"    <define name=\"calculation-method\">\n"
"        <element name=\"calculation-method\">\n"
"            <attribute name=\"name\"> <text/> </attribute>\n"
"            <optional>\n"
"                <attribute name=\"warning\"> <text/> </attribute>\n"
"            </optional>\n"
"            <optional>\n"
"                <element name=\"limits\">\n"
"                    <optional>\n"
"                        <element name=\"product-order\">\n"
"                            <data type=\"nonNegativeInteger\"/>\n"
"                        </element>\n"
"                    </optional>\n"
"                    <optional>\n"
"                        <element name=\"mission-time\"> <data type=\"double\"/> </element>\n"
"                    </optional>\n"
"                    <optional>\n"
"                        <element name=\"time-step\"> <data type=\"double\"/> </element>\n"
"                    </optional>\n"
"                    <optional>\n"
"                        <element name=\"cut-off\"> <ref name=\"probability-data\"/> </element>\n"
"                    </optional>\n"
"                    <optional>\n"
"                        <element name=\"number-of-sums\">\n"
"                            <data type=\"nonNegativeInteger\"/>\n"
"                        </element>\n"
"                    </optional>\n"
"                    <optional>\n"
"                        <element name=\"number-of-trials\">\n"
"                            <data type=\"nonNegativeInteger\"/>\n"
"                        </element>\n"
"                    </optional>\n"
"                    <optional>\n"
"                        <element name=\"seed\">\n"
"                            <data type=\"nonNegativeInteger\"/>\n"
"                        </element>\n"
"                    </optional>\n"
"                </element>\n"
"            </optional>\n"
"        </element>\n"
"    </define>\n"
"\n"
"    <!-- ############################################################### -->\n"
"    <!-- II. Results Layer -->\n"
"    <!-- ############################################################### -->\n"
"\n"
"    <define name=\"results-layer\">\n"
"        <element name=\"results\">\n"
"            <oneOrMore>\n"
"                <choice>\n"
"                    <ref name=\"sum-of-products\"/>\n"
"                    <ref name=\"importance\"/>\n"
"                    <ref name=\"safety-integrity-levels\"/>\n"
"                    <ref name=\"statistical-measure\"/>\n"
"                    <ref name=\"curve\"/>\n"
"                    <ref name=\"initiating-event\"/>\n"
"                </choice>\n"
"            </oneOrMore>\n"
"        </element>\n"
"    </define>\n"
"\n"
"    <!-- ============================================================= -->\n"
"    <!-- II.1. Sum of Products -->\n"
"    <!-- ============================================================= -->\n"
"\n"
"    <define name=\"sum-of-products\">\n"
"        <element name=\"sum-of-products\">\n"
"            <ref name=\"analysis-id\"/>\n"
"            <attribute name=\"basic-events\">\n"
"                <data type=\"nonNegativeInteger\"/>\n"
"            </attribute>\n"
"            <attribute name=\"products\">\n"
"                <data type=\"nonNegativeInteger\"/>\n"
"            </attribute>\n"
"            <optional>\n"
"                <attribute name=\"probability\"> <ref name=\"probability-data\"/> </attribute>\n"
"            </optional>\n"
"            <optional>\n"
"                <attribute name=\"distribution\">\n"
"                    <list>\n"
"                        <oneOrMore>\n"
"                            <data type=\"nonNegativeInteger\"/>\n"
"                        </oneOrMore>\n"
"                    </list>\n"
"                </attribute>\n"
"            </optional>\n"
"            <zeroOrMore>\n"
"                <ref name=\"product\"/>\n"
"            </zeroOrMore>\n"
"        </element>\n"
"    </define>\n"
"\n"
"    <define name=\"product\">\n"
"        <element name=\"product\">\n"
"            <attribute name=\"order\">\n"
"                <data type=\"positiveInteger\"/>\n"
"            </attribute>\n"
"            <optional>\n"
"                <attribute name=\"probability\"> <ref name=\"probability-data\"/> </attribute>\n"
"            </optional>\n"
"            <optional>\n"
"                <attribute name=\"contribution\"> <ref name=\"probability-data\"/> </attribute>\n"
"            </optional>\n"
"            <zeroOrMore>\n"
"                <ref name=\"literal\"/>\n"
"            </zeroOrMore>\n"
"        </element>\n"
"    </define>\n"
"\n"
"    <define name=\"literal\">\n"
"        <choice>\n"
"            <ref name=\"literal-event\"/>\n"
"            <element name=\"not\">\n"
"                <ref name=\"literal-event\"/>\n"
"            </element>\n"
"        </choice>\n"
"    </define>\n"
"\n"
"    <define name=\"literal-event\">\n"
"        <choice>\n"
"            <element name=\"basic-event\">\n"
"                <attribute name=\"name\"> <data type=\"NCName\"/> </attribute>\n"
"            </element>\n"
"            <element name=\"ccf-event\">\n"
"                <attribute name=\"ccf-group\"> <data type=\"NCName\"/> </attribute>\n"
"                <attribute name=\"order\"> <data type=\"positiveInteger\"/> </attribute>\n"
"                <attribute name=\"group-size\">\n"
"                    <data type=\"positiveInteger\"/>\n"
"                </attribute>\n"
"                <oneOrMore>\n"
"                    <element name=\"basic-event\">\n"
"                        <attribute name=\"name\"> <data type=\"NCName\"/> </attribute>\n"
"                    </element>\n"
"                </oneOrMore>\n"
"            </element>\n"
"        </choice>\n"
"    </define>\n"
"\n"
"    <!-- ============================================================= -->\n"
"    <!-- II.2. Statistical Measures -->\n"
"    <!-- ============================================================= -->\n"
"\n"
"    <define name=\"statistical-measure\">\n"
"        <element name=\"measure\">\n"
"            <ref name=\"analysis-id\"/>\n"
"            <element name=\"mean\">\n"
"                <attribute name=\"value\"> <ref name=\"probability-data\"/> </attribute>\n"
"            </element>\n"
"            <element name=\"standard-deviation\">\n"
"                <attribute name=\"value\"> <ref name=\"probability-data\"/> </attribute>\n"
"            </element>\n"
"            <element name=\"confidence-range\">\n"
"                <attribute name=\"percentage\">\n"
"                    <data type=\"double\">\n"
"                        <param name=\"minExclusive\">0</param>\n"
"                        <param name=\"maxExclusive\">100</param>\n"
"                    </data>\n"
"                </attribute>\n"
"                <attribute name=\"lower-bound\"> <ref name=\"probability-data\"/> </attribute>\n"
"                <attribute name=\"upper-bound\"> <ref name=\"probability-data\"/> </attribute>\n"
"            </element>\n"
"            <element name=\"error-factor\">\n"
"                <attribute name=\"percentage\">\n"
"                    <data type=\"double\">\n"
"                        <param name=\"minExclusive\">0</param>\n"
"                        <param name=\"maxExclusive\">100</param>\n"
"                    </data>\n"
"                </attribute>\n"
"                <attribute name=\"value\">\n"
"                    <data type=\"double\">\n"
"                        <param name=\"minExclusive\">0</param>\n"
"                    </data>\n"
"                </attribute>\n"
"            </element>\n"
"            <ref name=\"quantiles\"/>\n"
"            <ref name=\"histogram\"/>\n"
"        </element>\n"
"    </define>\n"
"\n"
"    <define name=\"quantiles\">\n"
"        <element name=\"quantiles\">\n"
"            <attribute name=\"number\"> <data type=\"positiveInteger\"/> </attribute>\n"
"            <oneOrMore>\n"
"                <ref name=\"quantile\"/>\n"
"            </oneOrMore>\n"
"        </element>\n"
"    </define>\n"
"\n"
"    <define name=\"quantile\">\n"
"        <element name=\"quantile\">\n"
"            <ref name=\"bin-data\"/>\n"
"        </element>\n"
"    </define>\n"
"\n"
"    <define name=\"histogram\">\n"
"        <element name=\"histogram\">\n"
"            <attribute name=\"number\"> <data type=\"positiveInteger\"/> </attribute>\n"
"            <oneOrMore>\n"
"                <ref name=\"bin\"/>\n"
"            </oneOrMore>\n"
"        </element>\n"
"    </define>\n"
"\n"
"    <define name=\"bin\">\n"
"        <element name=\"bin\">\n"
"            <ref name=\"bin-data\"/>\n"
"        </element>\n"
"    </define>\n"
"\n"
"    <define name=\"bin-data\">\n"
"        <attribute name=\"number\"> <data type=\"positiveInteger\"/> </attribute>\n"
"        <attribute name=\"value\"> <data type=\"double\"/> </attribute>\n"
"        <attribute name=\"lower-bound\"> <data type=\"double\"/> </attribute>\n"
"        <attribute name=\"upper-bound\"> <data type=\"double\"/> </attribute>\n"
"    </define>\n"
"\n"
"    <!-- ============================================================= -->\n"
"    <!-- II.3. Curves -->\n"
"    <!-- ============================================================= -->\n"
"\n"
"    <define name=\"curve\">\n"
"        <element name=\"curve\">\n"
"            <ref name=\"analysis-id\"/>\n"
"            <attribute name=\"X-title\"> <data type=\"string\"/> </attribute>\n"
"            <attribute name=\"Y-title\"> <data type=\"string\"/> </attribute>\n"
"            <optional>\n"
"                <attribute name=\"Z-title\"> <data type=\"string\"/> </attribute>\n"
"            </optional>\n"
"            <optional>\n"
"                <attribute name=\"X-unit\"> <ref name=\"unit\"/> </attribute>\n"
"            </optional>\n"
"            <optional>\n"
"                <attribute name=\"Y-unit\"> <ref name=\"unit\"/> </attribute>\n"
"            </optional>\n"
"            <optional>\n"
"                <attribute name=\"Z-unit\"> <ref name=\"unit\"/> </attribute>\n"
"            </optional>\n"
"            <zeroOrMore>\n"
"                <element name=\"point\">\n"
"                    <attribute name=\"X\"> <data type=\"double\"/> </attribute>\n"
"                    <attribute name=\"Y\"> <data type=\"double\"/> </attribute>\n"
"                    <optional>\n"
"                        <attribute name=\"Z\"> <data type=\"double\"/> </attribute>\n"
"                    </optional>\n"
"                </element>\n"
"            </zeroOrMore>\n"
"        </element>\n"
"    </define>\n"
"\n"
"    <define name=\"unit\">\n"
"        <choice>\n"
"            <value>seconds</value>\n"
"            <value>hours</value>\n"
"            <value>seconds-1</value>\n"
"            <value>hours-1</value>\n"
"            <value>years</value>\n"
"            <value>years-1</value>\n"
"        </choice>\n"
"    </define>\n"
"\n"
"    <!-- ============================================================= -->\n"
"    <!-- II.4. Importance -->\n"
"    <!-- ============================================================= -->\n"
"\n"
"    <define name=\"importance\">\n"
"        <element name=\"importance\">\n"
"            <ref name=\"analysis-id\"/>\n"
"            <attribute name=\"basic-events\">\n"
"                <data type=\"nonNegativeInteger\"/>\n"
"            </attribute>\n"
"            <zeroOrMore>\n"
"                <choice>\n"
"                    <element name=\"basic-event\">\n"
"                        <attribute name=\"name\"> <data type=\"NCName\"/> </attribute>\n"
"                        <ref name=\"importance-factors\"/>\n"
"                    </element>\n"
"                    <element name=\"ccf-event\">\n"
"                        <attribute name=\"ccf-group\"> <data type=\"NCName\"/> </attribute>\n"
"                        <attribute name=\"order\">\n"
"                            <data type=\"positiveInteger\"/>\n"
"                        </attribute>\n"
"                        <attribute name=\"group-size\">\n"
"                            <data type=\"positiveInteger\"/>\n"
"                        </attribute>\n"
"                        <ref name=\"importance-factors\"/>\n"
"                        <oneOrMore>\n"
"                            <element name=\"basic-event\">\n"
"                                <attribute name=\"name\"> <data type=\"NCName\"/> </attribute>\n"
"                            </element>\n"
"                        </oneOrMore>\n"
"                    </element>\n"
"                </choice>\n"
"            </zeroOrMore>\n"
"        </element>\n"
"    </define>\n"
"\n"
"    <define name=\"importance-factors\">\n"
"        <attribute name=\"occurrence\"> <data type=\"nonNegativeInteger\"/> </attribute>\n"
"        <attribute name=\"probability\"> <data type=\"double\"/> </attribute>\n"
"        <attribute name=\"DIF\"> <data type=\"double\"/> </attribute>\n"
"        <attribute name=\"MIF\"> <data type=\"double\"/> </attribute>\n"
"        <attribute name=\"CIF\"> <data type=\"double\"/> </attribute>\n"
"        <attribute name=\"RRW\"> <data type=\"double\"/> </attribute>\n"
"        <attribute name=\"RAW\"> <data type=\"double\"/> </attribute>\n"
"    </define>\n"
"\n"
"    <!-- ============================================================= -->\n"
"    <!-- II.5. Safety Integrity Levels -->\n"
"    <!-- ============================================================= -->\n"
"\n"
"    <define name=\"safety-integrity-levels\">\n"
"        <element name=\"safety-integrity-levels\">\n"
"            <ref name=\"analysis-id\"/>\n"
"            <attribute name=\"PFD-avg\"> <ref name=\"probability-data\"/> </attribute>\n"
"            <attribute name=\"PFH-avg\"> <ref name=\"probability-data\"/> </attribute>\n"
"            <ref name=\"histogram\"/>  <!-- Implicit PFD histogram -->\n"
"            <ref name=\"histogram\"/>  <!-- Implicit PFH histogram -->\n"
"        </element>\n"
"    </define>\n"
"\n"
"    <!-- ============================================================= -->\n"
"    <!-- II.6. Initiating events and Sequences -->\n"
"    <!-- ============================================================= -->\n"
"\n"
"    <define name=\"initiating-event\">\n"
"        <element name=\"initiating-event\">\n"
"            <attribute name=\"name\"> <data type=\"NCName\"/> </attribute>\n"
"            <optional>\n"
"                <group>\n"
"                    <attribute name=\"alignment\"> <data type=\"NCName\"/> </attribute>\n"
"                    <attribute name=\"phase\"> <data type=\"NCName\"/> </attribute>\n"
"                </group>\n"
"            </optional>\n"
"            <optional>\n"
"                <attribute name=\"description\"> <text/> </attribute>\n"
"            </optional>\n"
"            <optional>\n"
"                <attribute name=\"warning\"> <text/> </attribute>\n"
"            </optional>\n"
"            <attribute name=\"sequences\">\n"
"                <data type=\"nonNegativeInteger\"/>\n"
"            </attribute>\n"
"            <oneOrMore>\n"
"                <element name=\"sequence\">\n"
"                    <attribute name=\"name\"> <data type=\"NCName\"/> </attribute>\n"
"                    <attribute name=\"value\"> <ref name=\"probability-data\"/> </attribute>\n"
"                </element>\n"
"            </oneOrMore>\n"
"        </element>\n"
"    </define>\n"
"\n"
"</grammar>\n"
"";

} // namespace scram::schemas
