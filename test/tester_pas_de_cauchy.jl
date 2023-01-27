function tester_pas_de_cauchy(affiche, Pas_De_Cauchy)
    
    @testset "Cas test g nul" begin
        g = [0; 0]
        H = [1 0 ; 0 1]
        delta = 1
        s, e = Pas_De_Cauchy(g,H,delta)
        @test s ≈ [0; 0]
        @test e == 0
    end
    @testset "Cas test H nul, a = 0, b != 0" begin
        g = [1; 1]
        H = [0 0 ; 0 0]
        delta = sqrt(2)
        s, e = Pas_De_Cauchy(g,H,delta)
        @test s ≈ -g
        @test e == 1
    end
    @testset "Cas test H < 0, a < 0, b != 0" begin
        g = [1; 1]
        H = [-1 0 ; 0 -1]
        delta = sqrt(2)
        s, e = Pas_De_Cauchy(g,H,delta)
        @test s ≈ -g
        @test e == 1
    end
    @testset "Cas test e = 1" begin
        g = [1; 1]
        H = [1 0 ; 0 1]
        delta = 0.5
        s, e = Pas_De_Cauchy(g,H,delta)
        @test s ≈ -[1/(2*sqrt(2)); 1/(2*sqrt(2))]
        @test e == 1
    end
    @testset "Cas test e = -1" begin
        g = [1; 1]
        H = [1 0 ; 0 1]
        delta = 2
        s, e = Pas_De_Cauchy(g,H,delta)
        @test s ≈ -g
        @test e == -1
    end
end
