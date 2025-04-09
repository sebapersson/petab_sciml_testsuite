#=
    Test that likelihood values for the hybrid tests are not drastically changed
=#

using PEtabSciMLTestsuite, Test, YAML

const LLH_VALUES = (
    "001" => 33.02909543616718,
    "002" => 33.30638150294612,
    "003" => 68.96494660053902,
    "004" => 33.30435713294598,
    "005" => 33.30638150294739,
    "006" => 33.30638150294612,
    "007" => 33.30638150295205,
    "008" => 33.30638150294612,
    "009" => 33.30435704677708,
    "010" => 33.02883354468138,
    "011" => 33.02909543616718,
    "012" => 33.02909535816728,
    "013" => 33.30638141677723,
    "014" => 33.30638150295139,
    "015" => 68.96501052725615,
    "016" => 33.02909543616718,
    "017" => 33.30638150294612,
    "018" => 33.30435713294598)

# Build all hybrid tests and check that likelihood is consistent
PEtabSciMLTestsuite.create_hybrid_tests()
for (test_case, ref_value) in LLH_VALUES
    path_solutions = joinpath(@__DIR__, "..", "test_cases", "hybrid", test_case, "solutions.yaml")
    solutions = YAML.load_file(path_solutions)
    @test solutions["llh"] ≈ ref_value
end
